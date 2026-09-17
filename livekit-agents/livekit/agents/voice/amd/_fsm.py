"""Pure AMD policy: (state, event) -> (next state, effects).

The coordinator owns turn identity, model tasks, and public events. The FSM only
tracks call stages and pending work. All times use the same monotonic clock.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto

from .events import AMDCategory, AMDReason

ALLOWED = {
    AMDCategory.UNCERTAIN: frozenset(AMDCategory),
    AMDCategory.MACHINE_SCREENING: frozenset(
        {
            AMDCategory.MACHINE_SCREENING,
            AMDCategory.HUMAN,
            AMDCategory.MACHINE_VM,
            AMDCategory.MACHINE_UNAVAILABLE,
        }
    ),
    AMDCategory.MACHINE_VM: frozenset(
        {
            AMDCategory.MACHINE_VM,
            AMDCategory.HUMAN,
            AMDCategory.MACHINE_IVR,
            AMDCategory.MACHINE_UNAVAILABLE,
        }
    ),
    AMDCategory.MACHINE_IVR: frozenset(
        {
            AMDCategory.MACHINE_IVR,
            AMDCategory.HUMAN,
            AMDCategory.MACHINE_VM,
            AMDCategory.MACHINE_UNAVAILABLE,
        }
    ),
}
TERMINAL = frozenset({AMDCategory.HUMAN, AMDCategory.MACHINE_UNAVAILABLE})
MACHINE = frozenset(
    {
        AMDCategory.MACHINE_SCREENING,
        AMDCategory.MACHINE_VM,
        AMDCategory.MACHINE_IVR,
        AMDCategory.MACHINE_UNAVAILABLE,
    }
)


class AMDLifecycle(Enum):
    INITIALIZED = auto()
    PENDING = auto()
    ACTIVE = auto()
    FINISHED = auto()


@dataclass(frozen=True)
class Options:
    idle_timeout: float = 10.0
    voicemail_idle_timeout: float = 60.0
    timeout: float = 120.0
    inference_timeout: float = 1.5
    machine_silence_threshold: float = 1.5
    max_uncertain_turns: int = 3
    max_inference_timeouts: int = 3


@dataclass(frozen=True)
class Prediction:
    category: AMDCategory
    reason: AMDReason


@dataclass(frozen=True)
class Classifying:
    deadline: float


@dataclass(frozen=True)
class Holding:
    prediction: Prediction
    release_at: float | None


class VoicemailReply(Enum):
    AVAILABLE = auto()
    RESERVED = auto()
    COMMITTED = auto()


@dataclass(frozen=True)
class State:
    lifecycle: AMDLifecycle = AMDLifecycle.INITIALIZED
    category: AMDCategory = AMDCategory.UNCERTAIN
    work: Classifying | Holding | None = None
    speaking: bool = False
    silence_since: float | None = None
    hard_deadline: float | None = None
    idle_deadline: float | None = None
    uncertain_turns: int = 0
    inference_timeouts: int = 0
    voicemail_reply: VoicemailReply = VoicemailReply.AVAILABLE
    voicemail_message_played: bool = False
    latest_category: AMDCategory | None = None
    previous_turn: AMDCategory | None = None
    previous_stage: AMDCategory | None = None
    completion_reason: AMDReason = AMDReason.CANCELLED

    @property
    def next_deadline(self) -> float | None:
        work_at = (
            self.work.deadline
            if isinstance(self.work, Classifying)
            else self.work.release_at
            if isinstance(self.work, Holding)
            else None
        )
        return min(
            (at for at in (self.hard_deadline, self.idle_deadline, work_at) if at is not None),
            default=None,
        )


class Signal(Enum):
    ENTER = auto()
    START = auto()
    SPEECH_STARTED = auto()
    INFERENCE_FAILED = auto()
    DEADLINE_REACHED = auto()
    REPLY_REQUESTED = auto()
    REPLY_COMMITTED = auto()
    VOICEMAIL_PLAYED = auto()


@dataclass(frozen=True)
class TurnCommitted:
    has_transcript: bool
    silence_since: float | None


@dataclass(frozen=True)
class SpeechEnded:
    silence_since: float


@dataclass(frozen=True)
class PredictionReceived:
    category: AMDCategory


@dataclass(frozen=True)
class ActivityChanged:
    session_busy: bool


@dataclass(frozen=True)
class Finish:
    reason: AMDReason


Event = Signal | TurnCommitted | SpeechEnded | PredictionReceived | ActivityChanged | Finish


class Action(Enum):
    CLASSIFY = auto()
    CANCEL_CLASSIFICATION = auto()
    REUSE_PENDING_PREDICTION = auto()
    EXTRACT_MENU = auto()
    COMPLETE = auto()
    TRACK_VOICEMAIL = auto()


@dataclass(frozen=True)
class ReleasePrediction:
    prediction: Prediction
    previous_turn: AMDCategory | None


@dataclass(frozen=True)
class ReplyDecision:
    allow: bool
    instructions_for: AMDCategory | None = None
    track_voicemail: bool = False


Effect = Action | ReleasePrediction | ReplyDecision


@dataclass(frozen=True)
class Transition:
    state: State
    effects: tuple[Effect, ...] = ()


def transition(state: State, event: Event, *, now: float, options: Options) -> Transition:
    match event:
        case Signal.ENTER:
            if state.lifecycle is not AMDLifecycle.INITIALIZED:
                raise RuntimeError("use a new AMD instance for each run")
            return Transition(replace(state, lifecycle=AMDLifecycle.PENDING))
        case Signal.START if state.lifecycle is AMDLifecycle.PENDING:
            return Transition(
                replace(state, lifecycle=AMDLifecycle.ACTIVE, hard_deadline=now + options.timeout)
            )
        case Finish(reason):
            return _finish(state, reason)
        case Signal.REPLY_REQUESTED:
            return _reply(state)

    if state.lifecycle is not AMDLifecycle.ACTIVE:
        return Transition(state)

    match event:
        case Signal.SPEECH_STARTED:
            work = (
                replace(state.work, release_at=None)
                if isinstance(state.work, Holding)
                else state.work
            )
            return Transition(
                replace(state, speaking=True, silence_since=None, idle_deadline=None, work=work)
            )
        case SpeechEnded(silence_since):
            return _resume_hold(
                replace(state, speaking=False, silence_since=silence_since), now, options
            )
        case TurnCommitted(has_transcript, silence_since):
            state = replace(
                state,
                idle_deadline=None,
                silence_since=silence_since,
                voicemail_reply=(
                    VoicemailReply.AVAILABLE
                    if state.voicemail_reply is VoicemailReply.RESERVED
                    else state.voicemail_reply
                ),
            )
            if has_transcript:
                return Transition(
                    replace(state, work=Classifying(now + options.inference_timeout)),
                    (Action.CANCEL_CLASSIFICATION, Action.CLASSIFY),
                )
            if state.work is not None:
                result = _resume_hold(state, now, options)
                return Transition(result.state, (Action.REUSE_PENDING_PREDICTION, *result.effects))
            return _settle(state, Prediction(state.category, AMDReason.REUSED), now, options)
        case PredictionReceived() | Signal.INFERENCE_FAILED:
            if not isinstance(state.work, Classifying):
                return Transition(state)
            # Check the clock here too: a result can run before an overdue timer callback.
            if state.next_deadline is not None and state.next_deadline <= now:
                return _expire(state, now, options)
            category = event.category if isinstance(event, PredictionReceived) else state.category
            category = state.category if category is AMDCategory.UNCERTAIN else category
            valid = isinstance(event, PredictionReceived) and category in ALLOWED[state.category]
            prediction = Prediction(
                category if valid else state.category,
                AMDReason.PREDICTION if valid else AMDReason.INFERENCE_ERROR,
            )
            if valid:
                state = replace(state, inference_timeouts=0)
            return _settle(state, prediction, now, options)
        case Signal.DEADLINE_REACHED:
            return _expire(state, now, options)
        case ActivityChanged(session_busy):
            idle = not session_busy and not state.speaking and state.work is None
            timeout = (
                options.voicemail_idle_timeout
                if state.category is AMDCategory.MACHINE_VM
                else options.idle_timeout
            )
            deadline = (
                (state.idle_deadline if state.idle_deadline is not None else now + timeout)
                if idle
                else None
            )
            return Transition(replace(state, idle_deadline=deadline))
        case Signal.REPLY_COMMITTED if state.voicemail_reply is VoicemailReply.RESERVED:
            return Transition(
                replace(state, voicemail_reply=VoicemailReply.COMMITTED), (Action.TRACK_VOICEMAIL,)
            )
        case Signal.VOICEMAIL_PLAYED:
            return Transition(replace(state, voicemail_message_played=True))
    return Transition(state)


def _finish(state: State, reason: AMDReason) -> Transition:
    if state.lifecycle is AMDLifecycle.FINISHED:
        return Transition(state)
    return Transition(
        replace(
            state,
            lifecycle=AMDLifecycle.FINISHED,
            completion_reason=reason,
            work=None,
            hard_deadline=None,
            idle_deadline=None,
        ),
        (Action.CANCEL_CLASSIFICATION, Action.COMPLETE),
    )


def _expire(state: State, now: float, options: Options) -> Transition:
    effects: list[Effect] = []
    while (at := state.next_deadline) is not None and at <= now:
        # Hard deadline wins ties, followed by idle and pending work.
        if at == state.hard_deadline:
            result = _finish(state, AMDReason.TIMEOUT)
        elif at == state.idle_deadline:
            result = _finish(state, AMDReason.IDLE_TIMEOUT)
        elif isinstance(state.work, Holding):
            result = _publish(state, state.work.prediction, options)
        else:
            state = replace(state, inference_timeouts=state.inference_timeouts + 1)
            result = _settle(
                state, Prediction(state.category, AMDReason.INFERENCE_TIMEOUT), now, options
            )
            effects.append(Action.CANCEL_CLASSIFICATION)
        state = result.state
        effects.extend(result.effects)
    return Transition(state, tuple(effects))


def _release_at(state: State, options: Options) -> float | None:
    if state.speaking or state.silence_since is None:
        return None
    return state.silence_since + options.machine_silence_threshold


def _resume_hold(state: State, now: float, options: Options) -> Transition:
    if not isinstance(state.work, Holding):
        return Transition(state)
    release_at = _release_at(state, options)
    if release_at is not None and release_at <= now:
        return _publish(state, state.work.prediction, options)
    return Transition(replace(state, work=replace(state.work, release_at=release_at)))


def _settle(state: State, prediction: Prediction, now: float, options: Options) -> Transition:
    if prediction.category in MACHINE and options.machine_silence_threshold > 0:
        release_at = _release_at(state, options)
        if release_at is None or now < release_at:
            return Transition(
                replace(state, work=Holding(prediction, release_at), idle_deadline=None)
            )
    return _publish(state, prediction, options)


def _publish(state: State, prediction: Prediction, options: Options) -> Transition:
    effects: tuple[Effect, ...] = (ReleasePrediction(prediction, state.latest_category),)
    state = replace(state, work=None)
    if prediction.reason is AMDReason.REUSED:
        return Transition(state, effects)
    previous = state.latest_category
    state = replace(state, latest_category=prediction.category)
    if prediction.reason is AMDReason.PREDICTION:
        if prediction.category != state.category:
            state = replace(
                state,
                previous_stage=state.category,
                category=prediction.category,
                idle_deadline=None,
                voicemail_reply=VoicemailReply.AVAILABLE,
            )
        state = replace(
            state,
            previous_turn=previous,
            uncertain_turns=state.uncertain_turns + 1
            if state.category is AMDCategory.UNCERTAIN
            else 0,
        )
        reason = (
            AMDReason.FINISHED
            if state.category in TERMINAL
            else AMDReason.MAX_UNCERTAIN_TURNS
            if state.uncertain_turns >= options.max_uncertain_turns
            else None
        )
        if reason is None and state.category is AMDCategory.MACHINE_IVR:
            effects += (Action.EXTRACT_MENU,)
    else:
        reason = (
            AMDReason.INFERENCE_TIMEOUT
            if prediction.reason is AMDReason.INFERENCE_TIMEOUT
            and state.inference_timeouts >= options.max_inference_timeouts
            else None
        )
    if reason is not None:
        result = _finish(state, reason)
        return Transition(result.state, (*effects, *result.effects))
    return Transition(state, effects)


def _reply(state: State) -> Transition:
    if state.lifecycle is AMDLifecycle.FINISHED:
        human_after_machine = (
            state.category is AMDCategory.HUMAN and state.previous_stage in MACHINE
        )
        return Transition(
            state,
            (
                ReplyDecision(
                    allow=state.category is not AMDCategory.MACHINE_UNAVAILABLE,
                    instructions_for=AMDCategory.HUMAN if human_after_machine else None,
                ),
            ),
        )
    if state.work is not None:
        return Transition(state, (ReplyDecision(allow=False),))
    if state.category is AMDCategory.MACHINE_VM:
        if state.voicemail_reply is not VoicemailReply.AVAILABLE:
            return Transition(state, (ReplyDecision(allow=False),))
        return Transition(
            replace(state, voicemail_reply=VoicemailReply.RESERVED),
            (ReplyDecision(allow=True, instructions_for=state.category, track_voicemail=True),),
        )
    return Transition(
        state,
        (
            ReplyDecision(
                allow=True,
                instructions_for=None
                if state.category is AMDCategory.UNCERTAIN
                else state.category,
            ),
        ),
    )
