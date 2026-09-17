from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from livekit import rtc

from ... import inference, llm, stt
from ..._exceptions import APIError
from ...log import logger
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import EventEmitter, aio, is_given
from ...utils.misc import is_cloud
from ...utils.participant import wait_for_participant_attribute, wait_for_track_publication
from . import _fsm, _inference
from ._fsm import AMDLifecycle, ReplyDecision
from ._transcription import AMDRacingSTT
from ._turns import AMDClassifyRequest, SpeechWindow, Turn, Turns
from .events import (
    AMDCategory,
    AMDCompletedEvent,
    AMDMenuObservedEvent,
    AMDPredictionEvent,
    AMDReason,
)

if TYPE_CHECKING:
    from .._turn_hooks import TurnHooks
    from ..agent import Agent
    from ..agent_session import AgentSession
    from ..events import (
        AgentFalseInterruptionEvent,
        AgentStateChangedEvent,
        SpeechCreatedEvent,
        UserInputTranscribedEvent,
        UserStateChangedEvent,
    )
    from ..speech_handle import SpeechHandle

_DEFAULT_MAX_INFERENCE_TIMEOUTS = 3  # Allow transient delays before abandoning detection.
_MACHINE_SILENCE_THRESHOLD = 1.5  # Wait for a pause before releasing machine predictions.
_MENU_TIMEOUT = 5.0  # Stop optional menu work before it becomes stale.

_DEFAULT_SCREENING_INSTRUCTIONS = (
    "Call state: automated call screening. Answer the screening assistant's latest "
    "prompt briefly. Use what you know about who you are and why you are calling. "
    "Then wait for its next prompt."
)
_DEFAULT_VOICEMAIL_INSTRUCTIONS = (
    "Call state: voicemail. Deliver one concise, self-contained message. "
    "State who you are, why you are calling, and the next step for the recipient."
)
_DEFAULT_IVR_INSTRUCTIONS = (
    "Call state: an automated phone menu. Use the participant's actual prompt and "
    "the purpose of this call to choose the next step. Use send_dtmf_events for an "
    "explicit keypad choice, or give a short spoken answer when requested. "
    "Do not invent a menu option. After the action, wait for the next prompt."
)
_DEFAULT_HUMAN_INSTRUCTIONS = (
    "Call state: a human has answered. The latest participant turn is from "
    "the human, not the automated system. Earlier automated prompts no longer "
    "apply. Respond to the human's latest message and continue the call normally. "
    "Do not resume a response to an earlier automated prompt."
)


@dataclass
class _AMDResources:
    """Resources that exist from ``__aenter__`` until completion."""

    agent: Agent
    llm: llm.LLM
    completion: asyncio.Future[AMDCompletedEvent]
    stt: AMDRacingSTT


@dataclass
class _AMDTurnHooks:
    _amd: AMD
    _turn_id: int | None = None
    _track_voicemail: bool = False

    def on_user_turn_committed(self, transcript: str, end_of_turn_delay: float | None) -> TurnHooks:
        return self._amd._on_user_turn_committed(transcript, end_of_turn_delay)

    def on_user_turn_completed(self) -> None:
        self._amd._update_idle()

    def on_reply_generation(
        self, tools: list[llm.Tool | llm.Toolset]
    ) -> list[llm.Tool | llm.Toolset]:
        return self._amd._on_reply_generation(tools)

    async def should_reply(self, chat_ctx: llm.ChatContext) -> bool:
        if self._turn_id is None:
            return self._amd._state.category is not AMDCategory.MACHINE_UNAVAILABLE
        decision = await self._amd._should_reply(self._turn_id, chat_ctx)
        self._track_voicemail = decision.track_voicemail
        return decision.allow

    def on_agent_turn_committed(self, handle: SpeechHandle) -> None:
        if self._track_voicemail:
            assert self._turn_id is not None
            self._amd._track_voicemail(self._turn_id, handle)


class AMD(EventEmitter[Literal["amd_prediction", "amd_completed", "amd_menu_observed"]]):
    """Multi-turn answering-machine detection.

    Start an AgentSession before entering AMD. Enter AMD before creating a SIP
    participant. execute() waits for completion, not the first machine prediction.
    The SDK owns classification and stage control. Models can use any provider.

    AMD uses the customer's current pipeline Agent. Realtime models and
    agent handoffs during AMD are not supported. Normal hooks, interruptions,
    playback, and StopResponse stay in AgentSession. Menu events are informational
    and never execute actions.

    Example:
        async with AMD(session) as amd:
            result = await amd.execute()

    Args:
        session: Started session whose participant audio and client-side EOT to use.
        llm: Classification and menu model. When omitted, use
            ``google/gemini-3.1-flash-lite`` if LiveKit Cloud credentials are
            available; otherwise use the current Agent's LLM. Pass None to
            always use the current Agent's LLM. A string selects a LiveKit
            Inference model. Calls use ``session.conn_options.llm_conn_options``.
            Supplied models stay open when AMD completes.
        stt: Optional second streaming STT model. When omitted, use
            ``cartesia/ink-whisper`` if LiveKit Cloud credentials are available;
            otherwise use only the session transcript. Pass None to always use
            only the session transcript. A string selects a LiveKit Inference model.
        participant_identity: Select the participant before placing an outbound call.
        wait_until_answered: Discard pre-answer audio from AMD and AgentSession
            when True. When False, listen to subscribed SIP early media.
        screening_instructions: Instructions for each screening turn.
        voicemail_instructions: Instructions for one message per voicemail stage.
        ivr_instructions: Instructions for each IVR turn. The SDK supplies a DTMF
            tool only to IVR reply generations.
        human_instructions: Temporary instructions for the first human turn after
            a machine stage. Skipped if no machine stage preceded the human.
        idle_timeout: Silent, inactive time before AMD completes outside voicemail.
        voicemail_idle_timeout: Silent, inactive time in voicemail. Starts after
            playback finishes, to allow a delayed post-message menu to arrive.
        timeout: Hard limit from the start of listening.
        inference_timeout: Prediction deadline per committed turn. A late result
            can update the stage, but cannot change a reply that already started.
        machine_silence_threshold: Continuous participant silence before a machine
            prediction releases the turn. Includes silence before and during
            classification. Human and initial uncertain predictions do not wait.
            Set to zero to disable the extra silence wait.
        max_uncertain_turns: Consecutive uncertain predictions before completion.
        max_inference_timeouts: Prediction timeouts before completion. A valid
            prediction resets the count.
    """

    _DEFAULT_LLM_MODEL: str = "google/gemini-3.1-flash-lite"
    _DEFAULT_STT_MODEL: str = "cartesia/ink-whisper"

    def __init__(
        self,
        session: AgentSession,
        *,
        llm: NotGivenOr[llm.LLM | str | None] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | str | None] = NOT_GIVEN,
        participant_identity: NotGivenOr[str] = NOT_GIVEN,
        wait_until_answered: bool = True,
        screening_instructions: str = _DEFAULT_SCREENING_INSTRUCTIONS,
        voicemail_instructions: str = _DEFAULT_VOICEMAIL_INSTRUCTIONS,
        ivr_instructions: str = _DEFAULT_IVR_INSTRUCTIONS,
        human_instructions: str = _DEFAULT_HUMAN_INSTRUCTIONS,
        idle_timeout: float = 10.0,
        voicemail_idle_timeout: float = 60.0,
        timeout: float = 120.0,
        inference_timeout: float = 1.5,
        machine_silence_threshold: float = _MACHINE_SILENCE_THRESHOLD,
        max_uncertain_turns: int = 3,
        max_inference_timeouts: int = _DEFAULT_MAX_INFERENCE_TIMEOUTS,
    ) -> None:
        super().__init__()
        if (
            min(idle_timeout, voicemail_idle_timeout, timeout, inference_timeout) <= 0
            or max_uncertain_turns < 1
        ):
            raise ValueError("amd timeouts and max_uncertain_turns must be positive")
        if max_inference_timeouts < 1:
            raise ValueError("max_inference_timeouts must be positive")
        if machine_silence_threshold < 0:
            raise ValueError("machine_silence_threshold must be non-negative")
        if not is_given(llm) or not is_given(stt):
            api_key = os.getenv("LIVEKIT_INFERENCE_API_KEY") or os.getenv("LIVEKIT_API_KEY")
            api_secret = os.getenv("LIVEKIT_INFERENCE_API_SECRET") or os.getenv(
                "LIVEKIT_API_SECRET"
            )
            auto_select = (
                is_cloud(os.getenv("LIVEKIT_URL", "")) and bool(api_key) and bool(api_secret)
            )
            if not is_given(llm):
                llm = self._DEFAULT_LLM_MODEL if auto_select else None
            if not is_given(stt):
                stt = self._DEFAULT_STT_MODEL if auto_select else None

        self._session = session
        self._turn_hooks = _AMDTurnHooks(self)
        self._owns_llm = isinstance(llm, str)
        self._llm = inference.LLM.from_model_string(llm) if isinstance(llm, str) else llm
        self._owns_stt = isinstance(stt, str)
        self._stt = inference.STT.from_model_string(stt) if isinstance(stt, str) else stt
        if self._stt is not None and not self._stt.capabilities.streaming:
            raise ValueError("amd requires a streaming STT")
        self._participant_identity = participant_identity
        self._wait_until_answered = wait_until_answered
        self._instructions = {
            AMDCategory.MACHINE_SCREENING: screening_instructions,
            AMDCategory.MACHINE_VM: voicemail_instructions,
            AMDCategory.MACHINE_IVR: ivr_instructions,
            AMDCategory.HUMAN: human_instructions,
        }
        self._options = _fsm.Options(
            idle_timeout=idle_timeout,
            voicemail_idle_timeout=voicemail_idle_timeout,
            inference_timeout=inference_timeout,
            machine_silence_threshold=machine_silence_threshold,
            max_uncertain_turns=max_uncertain_turns,
            max_inference_timeouts=max_inference_timeouts,
        )
        self._timeout = timeout
        self._hard_deadline: float | None = None
        self._state = _fsm.State()
        self._turns = Turns()
        self._user_speech = SpeechWindow()
        self._pending_turn: Turn | None = None
        self._latest: AMDPredictionEvent | None = None
        self._voicemail_turn_id: int | None = None
        self._event_queue: deque[tuple[_fsm.Event, Turn | None, float]] = deque()
        self._dispatching = False
        self._session_id = uuid.uuid4().hex
        self._control_prefix = f"amd_{uuid.uuid4().hex}_"
        self._voicemail_handle: SpeechHandle | None = None
        self._voicemail_audio_start = 0
        self._prediction_changed = asyncio.Event()
        self._run: _AMDResources | None = None
        self._classifier_task: asyncio.Task[None] | None = None
        self._menu_atask: asyncio.Task[None] | None = None
        self._finish_atask: asyncio.Task[None] | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._subscriptions: list[tuple[EventEmitter[Any], str, Callable[..., Any]]] = []
        self._timer: asyncio.TimerHandle | None = None
        self._speeches: set[SpeechHandle] = set()

    @property
    def _resources(self) -> _AMDResources:
        if self._run is None:
            raise RuntimeError("enter AMD before use")
        return self._run

    @property
    def lifecycle(self) -> AMDLifecycle:
        """Current lifecycle state of this AMD run."""
        return self._state.lifecycle

    async def __aenter__(self) -> AMD:
        activity = self._session._activity
        if activity is None:
            raise RuntimeError("start AgentSession before entering AMD")
        if isinstance(activity.llm, llm.RealtimeModel):
            raise ValueError("amd does not support realtime models")
        if self._session.amd:
            raise RuntimeError("amd is already active")
        if self._session.options.ivr_detection or self._session._ivr_activity is not None:
            raise ValueError("disable session-level ivr_detection when using AMD")
        model = self._llm if self._llm is not None else activity.llm
        if not isinstance(model, llm.LLM):
            raise ValueError("amd requires an LLM for classification")
        self._state = _fsm.transition(
            self._state, _fsm.Signal.ENTER, now=time.monotonic(), options=self._options
        ).state
        self._run = _AMDResources(
            agent=activity.agent,
            llm=model,
            completion=asyncio.get_running_loop().create_future(),
            stt=AMDRacingSTT(
                self._stt,
                self._session.conn_options.stt_conn_options,
            ),
        )
        try:
            self._session._amd = self
            self._session._turn_hooks = self._turn_hooks
            activity._pause_authorization()
            self._subscriptions = [
                (self._session, "user_state_changed", self._on_user_state_changed),
                (self._session, "user_input_transcribed", self._on_user_input_transcribed),
                (self._session, "speech_created", self._on_speech_created),
                (self._session, "agent_state_changed", self._on_agent_state_changed),
                (self._session, "agent_false_interruption", self._on_false_interruption),
            ]
            if self._session._room_io:
                room = self._session._room_io.room
                self._subscriptions.append(
                    (room, "participant_disconnected", self._on_disconnected)
                )
                if is_given(self._participant_identity):
                    self._session._room_io.set_participant(self._participant_identity)
            for emitter, event, handler in self._subscriptions:
                emitter.on(event, handler)
            self._spawn(self._setup_listening())
        except BaseException:
            await self.aclose()
            raise
        return self

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Run a background task that cleanup cancels and that ends the run if it fails."""
        try:
            task = asyncio.create_task(coro)
        except BaseException:
            coro.close()
            raise
        self._tasks.add(task)
        task.add_done_callback(self._on_task_done)
        return task

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("amd task failed", exc_info=error)
            self._finish(AMDReason.INTERNAL_ERROR)

    async def execute(self) -> AMDCompletedEvent:
        return await asyncio.shield(self._resources.completion)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self.lifecycle is AMDLifecycle.INITIALIZED:
            return
        self._finish(AMDReason.CANCELLED)
        if self._finish_atask is not None:
            await asyncio.shield(self._finish_atask)

    async def _setup_listening(self) -> None:
        try:
            if not self._session._room_io:
                self._start_listening()
                return
            room = self._session._room_io.room
            publication = await wait_for_track_publication(
                room=room,
                identity=self._participant_identity
                if is_given(self._participant_identity)
                else None,
                kind=rtc.TrackKind.KIND_AUDIO,
                wait_for_subscription=True,
            )
            publisher = next(
                (
                    p
                    for p in room.remote_participants.values()
                    if publication.sid in p.track_publications
                ),
                None,
            )
            if publisher is None:
                self._finish(AMDReason.PARTICIPANT_MISSING)
                return
            self._participant_identity = publisher.identity
            if (
                self._wait_until_answered
                and publisher.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
            ):
                await wait_for_participant_attribute(
                    room, identity=publisher.identity, attribute="sip.callStatus", value="active"
                )
            if self.lifecycle is not AMDLifecycle.FINISHED:
                self._start_listening()
        except RuntimeError:
            self._finish(AMDReason.PARTICIPANT_MISSING)

    def _start_listening(self) -> None:
        if self.lifecycle is not AMDLifecycle.PENDING:
            return
        self._hard_deadline = time.monotonic() + self._timeout
        self._dispatch(_fsm.Signal.START)
        if self._session.user_state == "speaking":
            self._user_speech.started(time.monotonic())
            self._dispatch(_fsm.Signal.SPEECH_STARTED)
        self._update_idle()
        logger.info("amd listening", extra={"session_id": self._session_id})

    def _on_disconnected(self, participant: rtc.RemoteParticipant) -> None:
        if participant.identity == self._participant_identity:
            self._finish(AMDReason.PARTICIPANT_DISCONNECTED)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self.lifecycle is AMDLifecycle.ACTIVE:
            self._resources.stt.push_audio(frame)

    def on_dtmf_event(self, digits: str) -> None:
        if self.lifecycle in {AMDLifecycle.PENDING, AMDLifecycle.ACTIVE}:
            self._turns.dtmf_sent(digits)

    def _on_user_state_changed(self, event: UserStateChangedEvent) -> None:
        if self.lifecycle is not AMDLifecycle.ACTIVE:
            return
        now = time.monotonic()
        # Convert the Unix speech boundary to an age, then anchor it to the FSM's
        # monotonic clock. This preserves speech timing when the event arrives late.
        delay = (
            max(0, time.time() - event.speech_timestamp)
            if event.speech_timestamp is not None
            else 0
        )
        if event.new_state == "speaking":
            if activity := self._session._activity:
                activity._pause_authorization()
            self._user_speech.started(now - delay)
            self._dispatch(_fsm.Signal.SPEECH_STARTED)
        elif event.old_state == "speaking":
            self._user_speech.ended(now - delay)
            self._dispatch(_fsm.SpeechEnded(now - delay))

    def _on_user_input_transcribed(self, event: UserInputTranscribedEvent) -> None:
        if self.lifecycle is AMDLifecycle.ACTIVE and event.is_final:
            self._resources.stt.push_transcript(event.transcript)

    def _on_user_turn_committed(
        self, transcript: str, end_of_turn_delay: float | None
    ) -> TurnHooks:
        if self.lifecycle is not AMDLifecycle.ACTIVE:
            return self._turn_hooks
        if activity := self._session._activity:
            activity._pause_authorization()
        turn_transcript = self._resources.stt.end_turn(transcript)
        now = time.monotonic()
        duration = self._user_speech.commit(now, end_of_turn_delay or 0)
        turn = self._turns.commit(turn_transcript, duration, now)
        self._voicemail_turn_id = None
        self._dispatch(
            _fsm.TurnCommitted(bool(turn_transcript.transcript), self._user_speech.silence_since),
            turn=turn,
        )
        return _AMDTurnHooks(self, turn.turn_id)

    def _dispatch(self, event: _fsm.Event, *, turn: Turn | None = None) -> tuple[_fsm.Effect, ...]:
        """Complete each transition's effects before handling events raised by listeners."""
        now = time.monotonic()
        if self._hard_deadline is not None and now >= self._hard_deadline:
            self._event_queue.append((_fsm.Finish(AMDReason.TIMEOUT), None, now))
        self._event_queue.append((event, turn, now))
        if self._dispatching:
            return ()
        self._dispatching = True
        effects: tuple[_fsm.Effect, ...] = ()
        notify_prediction = False
        update_idle = False
        try:
            while self._event_queue:
                event, turn, now = self._event_queue.popleft()
                result = _fsm.transition(self._state, event, now=now, options=self._options)
                self._state = result.state
                effects = result.effects
                notify_prediction |= isinstance(event, _fsm.TurnCommitted) or any(
                    isinstance(effect, _fsm.ReleasePrediction) or effect is _fsm.Action.COMPLETE
                    for effect in effects
                )
                update_idle |= event not in (
                    _fsm.Signal.REPLY_REQUESTED,
                    _fsm.Signal.REPLY_COMMITTED,
                    _fsm.Signal.VOICEMAIL_PLAYED,
                )
                # Reuse keeps the original turn's timing and transcript for the release.
                origin = self._pending_turn or turn
                if self._state.work is not None:
                    self._pending_turn = origin
                for effect in effects:
                    if effect is _fsm.Action.CANCEL_CLASSIFICATION:
                        self._cancel_classification()
                    elif effect is _fsm.Action.CLASSIFY:
                        assert turn is not None
                        self._pending_turn = origin = turn
                        request = self._turns.request(
                            turn,
                            stage=self._state.category,
                            allowed=sorted(_fsm.ALLOWED[self._state.category]),
                        )
                        if self._menu_atask is not None:
                            self._menu_atask.cancel()
                        self._classifier_task = self._spawn(self._classify(turn, request))
                    elif effect is _fsm.Action.REUSE_PENDING_PREDICTION:
                        assert turn is not None and origin is not None
                        turn.prediction = origin.prediction
                    elif isinstance(effect, _fsm.ReleasePrediction):
                        assert origin is not None
                        self._record_prediction(origin, effect, now)
                        self._pending_turn = None
                    elif effect is _fsm.Action.EXTRACT_MENU:
                        assert origin is not None
                        if self.lifecycle is AMDLifecycle.ACTIVE:
                            self._menu_atask = self._spawn(
                                self._extract_menu(origin.turn_id, origin.transcript.transcript)
                            )
                    elif effect is _fsm.Action.COMPLETE:
                        self._pending_turn = None
                        self._turns.clear_pending_dtmf()
        except Exception:
            logger.exception("amd transition failed")
            self._event_queue.clear()
            self._state = _fsm.transition(
                self._state,
                _fsm.Finish(AMDReason.INTERNAL_ERROR),
                now=time.monotonic(),
                options=self._options,
            ).state
            notify_prediction = update_idle = True
        finally:
            self._dispatching = False
        if self.lifecycle is AMDLifecycle.FINISHED:
            self._hard_deadline = None
        if notify_prediction:
            self._prediction_changed.set()
        if update_idle:
            self._update_idle()
        if self.lifecycle is AMDLifecycle.FINISHED and self._finish_atask is None:
            self._finish_atask = asyncio.create_task(self._cleanup())
        return effects

    def _cancel_classification(self) -> None:
        task, self._classifier_task = self._classifier_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    def _record_prediction(self, turn: Turn, effect: _fsm.ReleasePrediction, now: float) -> None:
        prediction = effect.prediction
        event = AMDPredictionEvent(
            turn_id=turn.turn_id,
            category=prediction.category,
            reason=prediction.reason,
            transcript=turn.transcript.transcript,
            speech_duration=turn.speech_duration,
            delay=now - turn.committed_at,
            inference_duration=turn.inference_duration
            if prediction.reason is AMDReason.PREDICTION
            else None,
            prev_turn_category=effect.previous_turn,
            prev_stage_category=self._state.previous_stage,
            voicemail_message_played=self._state.voicemail_message_played,
        )
        turn.prediction.result = event
        if prediction.reason is not AMDReason.REUSED:
            self._latest = event
            self._release_prediction(event.model_copy())

    def _release_prediction(self, event: AMDPredictionEvent) -> None:
        if event.state_changed and (activity := self._session._activity):
            activity._cancel_preemptive_generation()
        self.emit("amd_prediction", event)
        self._session._on_amd_prediction(event)
        logger.info(
            "amd prediction",
            extra={
                "turn_id": event.turn_id,
                "category": event.category.value,
                "reason": event.reason,
                "session_id": self._session_id,
            },
        )

    async def _classify(self, turn: Turn, request: AMDClassifyRequest) -> None:
        started = time.monotonic()
        event: _fsm.Event
        try:
            result = await _inference.classify(
                self._resources.llm,
                request,
                conn_options=self._session.conn_options.llm_conn_options,
            )
        except Exception as exc:
            if asyncio.current_task() is not self._classifier_task:
                return
            if not isinstance(exc, (APIError, ValueError, asyncio.TimeoutError)):
                raise
            logger.warning(
                "amd classification failed",
                extra={"turn_id": turn.turn_id, "error_type": type(exc).__name__},
            )
            event = _fsm.Signal.INFERENCE_FAILED
        else:
            turn.inference_duration = time.monotonic() - started
            event = _fsm.PredictionReceived(result.category)
        # Cancellation is best effort. A provider may still return after timeout or supersession.
        if asyncio.current_task() is self._classifier_task:
            self._dispatch(event)

    async def _extract_menu(self, turn_id: int, transcript: str) -> None:
        started = time.monotonic()
        try:
            menu = await asyncio.wait_for(
                _inference.extract_ivr_menu(
                    self._resources.llm,
                    transcript,
                    conn_options=self._session.conn_options.llm_conn_options,
                ),
                _MENU_TIMEOUT,
            )
        except Exception as exc:
            logger.debug("amd menu extraction failed", extra={"error_type": type(exc).__name__})
            return
        if self.lifecycle is AMDLifecycle.FINISHED or not (menu.menu or menu.options):
            return
        self.emit(
            "amd_menu_observed",
            AMDMenuObservedEvent(
                session_id=self._session_id,
                turn_id=turn_id,
                menu=menu.menu,
                options=menu.options,
                extraction_duration=time.monotonic() - started,
            ),
        )

    async def _wait_for_prediction(self, turn_id: int) -> None:
        """Wait until the turn has a prediction, a newer turn replaced it, or the run finished."""
        while (
            turn_id == self._turns.turn_id
            and self.lifecycle is not AMDLifecycle.FINISHED
            and self._turns.prediction(turn_id) is None
        ):
            self._prediction_changed.clear()
            await self._prediction_changed.wait()

    async def _should_reply(self, turn_id: int, chat_ctx: llm.ChatContext) -> ReplyDecision:
        """Wait for the turn's prediction, then add stage instructions when a reply is allowed."""
        if turn_id in self._turns:
            await self._wait_for_prediction(turn_id)
            if (
                self.lifecycle is not AMDLifecycle.FINISHED
                and self._session.current_agent is not self._resources.agent
            ):
                self._finish(AMDReason.AGENT_CHANGED)
                return ReplyDecision(allow=False)
        if (
            self.lifecycle is AMDLifecycle.FINISHED
            and self._state.category is AMDCategory.MACHINE_UNAVAILABLE
        ):
            return ReplyDecision(allow=False)
        if turn_id not in self._turns:
            return ReplyDecision(allow=True)
        if turn_id != self._turns.turn_id:
            return ReplyDecision(allow=False)
        effects = self._dispatch(_fsm.Signal.REPLY_REQUESTED)
        reply_decision = next(effect for effect in effects if isinstance(effect, ReplyDecision))
        if reply_decision.track_voicemail:
            self._voicemail_turn_id = turn_id
        if not reply_decision.allow:
            return reply_decision
        if reply_decision.instructions_for is not None:
            self._add_instructions(chat_ctx, reply_decision.instructions_for, turn_id)
        if (
            turn_id in self._turns
            and self.lifecycle is AMDLifecycle.ACTIVE
            and (activity := self._session._activity)
        ):
            activity._resume_authorization()
        return reply_decision

    def _add_instructions(
        self, chat_ctx: llm.ChatContext, stage: AMDCategory, turn_id: int
    ) -> None:
        content = self._instructions[stage]
        if stage is AMDCategory.MACHINE_IVR and self._state.voicemail_message_played:
            content += "\nThe voicemail message already played locally."
        chat_ctx.add_message(
            id=f"{self._control_prefix}{turn_id}",
            role="user",
            content=content,
            extra={"amd_run": self._session_id, "amd_stage": self._state.category.value},
        )

    def _on_reply_generation(
        self, tools: list[llm.Tool | llm.Toolset]
    ) -> list[llm.Tool | llm.Toolset]:
        if (
            self.lifecycle is AMDLifecycle.ACTIVE
            and self._state.category == AMDCategory.MACHINE_IVR
        ):
            from ...beta.tools.send_dtmf import send_dtmf_events

            if not any(tool.id == send_dtmf_events.id for tool in tools):
                return [*tools, send_dtmf_events]
        return tools

    # region: track voicemail speech handle
    def _track_voicemail(self, turn_id: int, handle: SpeechHandle) -> None:
        if self._voicemail_turn_id != turn_id:
            return
        if _fsm.Action.TRACK_VOICEMAIL not in self._dispatch(_fsm.Signal.REPLY_COMMITTED):
            return
        self._voicemail_turn_id = None
        if self._voicemail_handle is not None:
            self._voicemail_handle.remove_done_callback(self._on_voicemail_done)
        self._voicemail_handle = handle
        output = self._session.output.audio
        self._voicemail_audio_start = output.captured_playout_segments if output else 0
        handle.add_done_callback(self._on_voicemail_done)

    def _on_voicemail_done(self, handle: SpeechHandle) -> None:
        if handle is not self._voicemail_handle:
            return
        handle.remove_done_callback(self._on_voicemail_done)
        self._voicemail_handle = None
        if (
            self.lifecycle is AMDLifecycle.FINISHED
            or handle.interrupted
            or handle.exception() is not None
        ):
            return
        output = self._session.output.audio
        if output and output.captured_playout_segments > self._voicemail_audio_start:
            self._dispatch(_fsm.Signal.VOICEMAIL_PLAYED)

    def _on_speech_created(self, event: SpeechCreatedEvent) -> None:
        if self.lifecycle is AMDLifecycle.FINISHED:
            return
        self._speeches.add(event.speech_handle)
        event.speech_handle.add_done_callback(self._on_speech_done)
        self._update_idle()

    def _on_speech_done(self, handle: SpeechHandle) -> None:
        self._speeches.discard(handle)
        self._update_idle()

    # endregion

    def _on_agent_state_changed(self, event: AgentStateChangedEvent) -> None:
        self._update_idle()

    def _on_false_interruption(self, event: AgentFalseInterruptionEvent) -> None:
        # AgentSession clears its pause state after it emits this event.
        asyncio.get_running_loop().call_soon(self._update_idle)

    def _on_deadline(self) -> None:
        self._dispatch(_fsm.Signal.DEADLINE_REACHED)

    def _update_idle(self) -> None:
        activity = self._session._activity
        self._state = _fsm.transition(
            self._state,
            _fsm.ActivityChanged(
                bool(self._speeches) or activity is None or activity._is_agent_active
            ),
            now=time.monotonic(),
            options=self._options,
        ).state
        self._arm_timer()

    @property
    def _next_deadline(self) -> float | None:
        return min(
            (at for at in (self._hard_deadline, self._state.next_deadline) if at is not None),
            default=None,
        )

    def _arm_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        if (at := self._next_deadline) is not None:
            self._timer = asyncio.get_running_loop().call_later(
                max(0, at - time.monotonic()), self._on_deadline
            )

    def _finish(self, reason: AMDReason) -> None:
        self._dispatch(_fsm.Finish(reason))

    def _completion(self) -> AMDCompletedEvent:
        if self.lifecycle is not AMDLifecycle.FINISHED:
            raise RuntimeError("AMD has not completed")
        return AMDCompletedEvent(
            category=self._state.category,
            reason=self._state.completion_reason,
            turn_id=self._latest.turn_id if self._latest else self._turns.turn_id,
            transcript=self._latest.transcript if self._latest else "",
            prev_turn_category=self._state.previous_turn,
            prev_stage_category=self._state.previous_stage,
            voicemail_message_played=self._state.voicemail_message_played,
        )

    async def _cleanup(self) -> None:
        try:
            for emitter, event, handler in self._subscriptions:
                emitter.off(event, handler)
            for speech in self._speeches:
                speech.remove_done_callback(self._on_speech_done)
            if self._voicemail_handle is not None:
                self._voicemail_handle.remove_done_callback(self._on_voicemail_done)
                self._voicemail_handle = None
            await aio.cancel_and_wait(*self._tasks)
            if activity := self._session._activity:
                if self._state.category == AMDCategory.MACHINE_UNAVAILABLE:
                    activity._cancel_pending_speeches()
                else:
                    activity._cancel_preemptive_generation()
                activity._resume_authorization()
            if self._session._amd is self:
                self._session._amd = None
                self._session._turn_hooks = None
            close_tasks = [self._resources.stt.aclose()]
            if self._owns_stt and self._stt is not None:
                close_tasks.append(self._stt.aclose())
            if self._owns_llm and self._llm is not None:
                close_tasks.append(self._llm.aclose())
            for error in await asyncio.gather(*close_tasks, return_exceptions=True):
                if isinstance(error, BaseException):
                    logger.warning(
                        "amd resource cleanup failed", extra={"error_type": type(error).__name__}
                    )
        finally:
            result = self._completion()
            self._resources.completion.set_result(result)
            self.emit("amd_completed", result)
