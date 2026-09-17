from __future__ import annotations

from dataclasses import replace

import pytest

from livekit.agents.voice.amd import _fsm as fsm
from livekit.agents.voice.amd.events import AMDCategory as Category, AMDReason as Reason

pytestmark = pytest.mark.unit

OPTIONS = fsm.Options(inference_timeout=1, machine_silence_threshold=0)
ACTIVE = fsm.State(lifecycle=fsm.AMDLifecycle.ACTIVE, hard_deadline=120)


def send(
    state: fsm.State, event: fsm.Event, now: float = 0, *, options: fsm.Options = OPTIONS
) -> fsm.Transition:
    return fsm.transition(state, event, now=now, options=options)


def request(state: fsm.State, now: float = 0) -> fsm.State:
    result = send(state, fsm.TurnCommitted(True, now), now)
    assert result.effects == (fsm.Action.CANCEL_CLASSIFICATION, fsm.Action.CLASSIFY)
    return result.state


def released(result: fsm.Transition) -> fsm.Prediction:
    return next(e.prediction for e in result.effects if isinstance(e, fsm.ReleasePrediction))


def stage(category: Category) -> fsm.State:
    return send(request(ACTIVE), fsm.PredictionReceived(category), 0.1).state


def test_transition_is_repeatable_and_does_not_mutate_input() -> None:
    state = request(ACTIVE)
    event = fsm.PredictionReceived(Category.MACHINE_VM)
    first = send(state, event, 0.1)
    assert send(state, event, 0.1) == first
    assert state == replace(ACTIVE, work=fsm.Classifying(1), silence_since=0)
    assert first.state.category is Category.MACHINE_VM
    assert first.state.work is None


def test_lifecycle_and_fixed_hard_deadline() -> None:
    initial = fsm.State()
    pending = send(initial, fsm.Signal.ENTER).state
    assert pending.lifecycle is fsm.AMDLifecycle.PENDING
    assert send(pending, fsm.ActivityChanged(False), 3).state.next_deadline is None
    state = send(pending, fsm.Signal.START, 5).state
    assert state.hard_deadline == 125
    assert send(state, fsm.Signal.START, 6).state == state
    state = send(request(state, 7), fsm.PredictionReceived(Category.MACHINE_VM), 7.1).state
    state = send(state, fsm.ActivityChanged(False), 10).state
    assert state.next_deadline == 70
    state = send(state, fsm.ActivityChanged(True), 69).state
    assert state.next_deadline == 125
    assert send(state, fsm.Signal.DEADLINE_REACHED, 124).state.lifecycle is fsm.AMDLifecycle.ACTIVE
    result = send(state, fsm.Signal.DEADLINE_REACHED, 125)
    assert result.state.lifecycle is fsm.AMDLifecycle.FINISHED
    assert result.state.next_deadline is None
    assert result.state.completion_reason is Reason.TIMEOUT
    assert result.state.category is Category.MACHINE_VM
    with pytest.raises(RuntimeError, match="new AMD instance"):
        send(result.state, fsm.Signal.ENTER)
    idle = send(ACTIVE, fsm.ActivityChanged(False)).state
    assert (
        send(idle, fsm.Signal.DEADLINE_REACHED, 121).state.completion_reason is Reason.IDLE_TIMEOUT
    )


@pytest.mark.parametrize("committed_at", [118, 119, 119.5])
def test_delayed_timer_orders_inference_and_hard_deadline(committed_at: float) -> None:
    result = send(request(ACTIVE, committed_at), fsm.Signal.DEADLINE_REACHED, 121)
    assert result.state.completion_reason is Reason.TIMEOUT
    predictions = [e.prediction for e in result.effects if isinstance(e, fsm.ReleasePrediction)]
    assert predictions == (
        [fsm.Prediction(Category.UNCERTAIN, Reason.INFERENCE_TIMEOUT)]
        if committed_at == 118
        else []
    )
    assert result.state.next_deadline is None


@pytest.mark.parametrize("committed_at", [118, 118.5, 119])
def test_delayed_timer_orders_silence_release_and_hard_deadline(committed_at: float) -> None:
    options = replace(OPTIONS, machine_silence_threshold=1.5)
    result = send(
        request(ACTIVE, committed_at),
        fsm.PredictionReceived(Category.MACHINE_UNAVAILABLE),
        committed_at + 0.1,
        options=options,
    )
    assert result.effects == ()
    result = send(result.state, fsm.Signal.DEADLINE_REACHED, 121, options=options)
    assert result.state.completion_reason is (
        Reason.FINISHED if committed_at == 118 else Reason.TIMEOUT
    )
    assert result.state.next_deadline is None


@pytest.mark.parametrize(
    "current",
    [Category.UNCERTAIN, Category.MACHINE_SCREENING, Category.MACHINE_VM, Category.MACHINE_IVR],
)
@pytest.mark.parametrize("category", list(Category))
def test_stage_transitions(current: Category, category: Category) -> None:
    allowed = {
        Category.UNCERTAIN: set(Category),
        Category.MACHINE_SCREENING: {
            Category.MACHINE_SCREENING,
            Category.HUMAN,
            Category.MACHINE_VM,
            Category.MACHINE_UNAVAILABLE,
        },
        Category.MACHINE_VM: {
            Category.MACHINE_VM,
            Category.HUMAN,
            Category.MACHINE_IVR,
            Category.MACHINE_UNAVAILABLE,
        },
        Category.MACHINE_IVR: {
            Category.MACHINE_IVR,
            Category.HUMAN,
            Category.MACHINE_VM,
            Category.MACHINE_UNAVAILABLE,
        },
    }
    state = stage(current)
    result = send(request(state, 1), fsm.PredictionReceived(category), 1.1)
    effective = current if category is Category.UNCERTAIN else category
    valid = effective in allowed[current]
    expected = effective if valid else current
    assert released(result) == fsm.Prediction(
        expected, Reason.PREDICTION if valid else Reason.INFERENCE_ERROR
    )
    assert result.state.category is expected
    assert (result.state.lifecycle is fsm.AMDLifecycle.FINISHED) == (
        expected in {Category.HUMAN, Category.MACHINE_UNAVAILABLE}
    )
    assert (fsm.Action.EXTRACT_MENU in result.effects) == (
        valid and expected is Category.MACHINE_IVR
    )
    assert result.state.previous_stage == (current if expected != current else state.previous_stage)


@pytest.mark.parametrize(
    "event",
    [
        fsm.Signal.INFERENCE_FAILED,
        fsm.Signal.DEADLINE_REACHED,
        fsm.TurnCommitted(False, 2),
        fsm.PredictionReceived(Category.MACHINE_SCREENING),
    ],
)
def test_fallbacks_and_reuse_do_not_extract_menus(event: fsm.Event) -> None:
    state = stage(Category.MACHINE_IVR)
    if not isinstance(event, fsm.TurnCommitted):
        state = request(state, 1)
    result = send(state, event, 2 if event is fsm.Signal.DEADLINE_REACHED else 1.1)
    assert fsm.Action.EXTRACT_MENU not in result.effects
    assert result.state.category is Category.MACHINE_IVR


def test_menu_is_extracted_only_after_silence_release() -> None:
    options = replace(OPTIONS, machine_silence_threshold=1.5)
    result = send(
        request(ACTIVE), fsm.PredictionReceived(Category.MACHINE_IVR), 0.1, options=options
    )
    assert result.effects == ()
    assert result.state.category is Category.UNCERTAIN
    result = send(result.state, fsm.Signal.DEADLINE_REACHED, 1.5, options=options)
    assert released(result).category is Category.MACHINE_IVR
    assert fsm.Action.EXTRACT_MENU in result.effects


@pytest.mark.parametrize("at", [1, 1.01, 3])
@pytest.mark.parametrize(
    "event", [fsm.PredictionReceived(Category.HUMAN), fsm.Signal.INFERENCE_FAILED]
)
def test_deadline_is_final_even_if_result_runs_before_timer(at: float, event: fsm.Event) -> None:
    state = request(ACTIVE)
    result = send(state, event, at)
    assert released(result) == fsm.Prediction(Category.UNCERTAIN, Reason.INFERENCE_TIMEOUT)
    assert fsm.Action.CANCEL_CLASSIFICATION in result.effects
    assert result.state.inference_timeouts == 1
    assert result.state.lifecycle is fsm.AMDLifecycle.ACTIVE
    assert result.state.work is None
    assert send(result.state, event, at + 1) == fsm.Transition(result.state)


def test_result_just_before_deadline_is_accepted() -> None:
    result = send(request(ACTIVE), fsm.PredictionReceived(Category.HUMAN), 0.999)
    assert released(result).reason is Reason.PREDICTION
    assert result.state.category is Category.HUMAN


@pytest.mark.parametrize("limit", [1, 3])
@pytest.mark.parametrize(
    "event",
    [
        fsm.Signal.INFERENCE_FAILED,
        fsm.PredictionReceived(Category.HUMAN),
        fsm.PredictionReceived(Category.MACHINE_IVR),
    ],
)
def test_late_result_cannot_replace_held_timeout(limit: int, event: fsm.Event) -> None:
    options = replace(OPTIONS, machine_silence_threshold=1.5, max_inference_timeouts=limit)
    result = send(
        request(stage(Category.MACHINE_SCREENING), 1),
        fsm.Signal.DEADLINE_REACHED,
        2,
        options=options,
    )
    assert result.effects == (fsm.Action.CANCEL_CLASSIFICATION,)
    assert isinstance(result.state.work, fsm.Holding)
    assert send(result.state, event, 2.1, options=options) == fsm.Transition(result.state)
    result = send(result.state, fsm.Signal.DEADLINE_REACHED, 2.5, options=options)
    assert released(result) == fsm.Prediction(Category.MACHINE_SCREENING, Reason.INFERENCE_TIMEOUT)
    assert (result.state.lifecycle is fsm.AMDLifecycle.FINISHED) == (limit == 1)


def test_valid_prediction_resets_timeout_count() -> None:
    options = replace(OPTIONS, max_inference_timeouts=2)
    state = send(request(ACTIVE), fsm.Signal.DEADLINE_REACHED, 1, options=options).state
    state = send(
        request(state, 2), fsm.PredictionReceived(Category.MACHINE_SCREENING), 2.1, options=options
    ).state
    assert state.inference_timeouts == 0
    state = send(request(state, 3), fsm.Signal.DEADLINE_REACHED, 4, options=options).state
    assert state.lifecycle is fsm.AMDLifecycle.ACTIVE
    state = send(request(state, 5), fsm.Signal.DEADLINE_REACHED, 6, options=options).state
    assert state.completion_reason is Reason.INFERENCE_TIMEOUT
    assert state.lifecycle is fsm.AMDLifecycle.FINISHED


def test_uncertain_limit_counts_model_predictions_only() -> None:
    state = ACTIVE
    for at in range(2):
        state = send(request(state, at), fsm.PredictionReceived(Category.UNCERTAIN), at + 0.1).state
    assert state.uncertain_turns == 2
    state = send(request(state, 2), fsm.Signal.INFERENCE_FAILED, 2.1).state
    state = send(state, fsm.TurnCommitted(False, 3), 3).state
    assert state.uncertain_turns == 2
    state = send(request(state, 4), fsm.PredictionReceived(Category.UNCERTAIN), 4.1).state
    assert state.completion_reason is Reason.MAX_UNCERTAIN_TURNS
    established = stage(Category.MACHINE_SCREENING)
    for at in range(1, 5):
        established = send(
            request(established, at), fsm.PredictionReceived(Category.UNCERTAIN), at + 0.1
        ).state
    assert established.lifecycle is fsm.AMDLifecycle.ACTIVE
    assert established.uncertain_turns == 0


@pytest.mark.parametrize("holding", [False, True])
def test_empty_turn_reuses_pending_prediction(holding: bool) -> None:
    options = replace(OPTIONS, machine_silence_threshold=1.5)
    state = request(ACTIVE)
    if holding:
        state = send(state, fsm.PredictionReceived(Category.MACHINE_VM), 0.1, options=options).state
    result = send(state, fsm.TurnCommitted(False, 0.5), 0.5, options=options)
    assert result.effects == (fsm.Action.REUSE_PENDING_PREDICTION,)
    assert result.state.next_deadline == (2 if holding else 1)
    if holding:
        result = send(result.state, fsm.Signal.DEADLINE_REACHED, 2, options=options)
    else:
        result = send(
            result.state, fsm.PredictionReceived(Category.MACHINE_VM), 0.6, options=options
        )
        result = send(result.state, fsm.Signal.DEADLINE_REACHED, 2, options=options)
    assert released(result).category is Category.MACHINE_VM


def test_empty_turn_without_pending_work_reuses_stage() -> None:
    result = send(stage(Category.MACHINE_SCREENING), fsm.TurnCommitted(False, 1), 1)
    assert released(result) == fsm.Prediction(Category.MACHINE_SCREENING, Reason.REUSED)
    assert fsm.Action.CLASSIFY not in result.effects


def test_new_transcript_supersedes_hold() -> None:
    options = replace(OPTIONS, machine_silence_threshold=1.5)
    state = send(
        request(ACTIVE), fsm.PredictionReceived(Category.MACHINE_VM), 0.1, options=options
    ).state
    state = request(state, 0.5)
    assert state.work == fsm.Classifying(1.5)
    result = send(state, fsm.PredictionReceived(Category.HUMAN), 0.6, options=options)
    assert result.state.category is Category.HUMAN
    assert released(result).category is Category.HUMAN


def test_speech_pauses_hold_and_end_rearms_without_a_commit() -> None:
    options = replace(OPTIONS, machine_silence_threshold=1.5)
    state = send(
        request(ACTIVE), fsm.PredictionReceived(Category.MACHINE_VM), 0.1, options=options
    ).state
    state = send(state, fsm.Signal.SPEECH_STARTED, 1, options=options).state
    assert isinstance(state.work, fsm.Holding) and state.work.release_at is None
    assert state.next_deadline == 120
    state = send(state, fsm.SpeechEnded(2), 2.1, options=options).state
    assert state.next_deadline == 3.5
    assert send(state, fsm.Signal.DEADLINE_REACHED, 3.4, options=options).effects == ()
    assert (
        released(send(state, fsm.Signal.DEADLINE_REACHED, 3.5, options=options)).category
        is Category.MACHINE_VM
    )


def test_idle_pauses_for_speech_and_pending_work() -> None:
    state = send(ACTIVE, fsm.ActivityChanged(False)).state
    assert state.idle_deadline == 10
    state = send(state, fsm.Signal.SPEECH_STARTED, 1).state
    assert send(state, fsm.ActivityChanged(False), 2).state.idle_deadline is None
    state = send(state, fsm.SpeechEnded(3), 3).state
    state = send(state, fsm.ActivityChanged(False), 3).state
    assert state.idle_deadline == 13
    state = request(state, 4)
    assert send(state, fsm.ActivityChanged(False), 4).state.idle_deadline is None


def test_voicemail_reservation_commit_and_playback_are_distinct() -> None:
    state = stage(Category.MACHINE_VM)
    result = send(state, fsm.Signal.REPLY_REQUESTED)
    assert result.effects == (fsm.ReplyDecision(True, Category.MACHINE_VM, True),)
    state = result.state
    assert state.voicemail_reply is fsm.VoicemailReply.RESERVED
    assert send(state, fsm.Signal.REPLY_REQUESTED).effects == (fsm.ReplyDecision(False),)
    result = send(state, fsm.Signal.REPLY_COMMITTED)
    assert result.effects == (fsm.Action.TRACK_VOICEMAIL,)
    assert result.state.voicemail_reply is fsm.VoicemailReply.COMMITTED
    assert not result.state.voicemail_message_played
    assert send(result.state, fsm.Signal.REPLY_COMMITTED).effects == ()
    state = send(result.state, fsm.Signal.VOICEMAIL_PLAYED).state
    assert state.voicemail_message_played
    state = send(request(state, 1), fsm.PredictionReceived(Category.MACHINE_IVR), 1.1).state
    state = send(request(state, 2), fsm.PredictionReceived(Category.MACHINE_VM), 2.1).state
    assert send(state, fsm.Signal.REPLY_REQUESTED).effects == (
        fsm.ReplyDecision(True, Category.MACHINE_VM, True),
    )
    assert state.voicemail_message_played


def test_new_turn_releases_uncommitted_voicemail_reservation() -> None:
    state = send(stage(Category.MACHINE_VM), fsm.Signal.REPLY_REQUESTED).state
    state = send(state, fsm.TurnCommitted(False, 1), 1).state
    assert state.voicemail_reply is fsm.VoicemailReply.AVAILABLE
    assert send(state, fsm.Signal.REPLY_COMMITTED).effects == ()
    assert send(state, fsm.Signal.REPLY_REQUESTED).effects == (
        fsm.ReplyDecision(True, Category.MACHINE_VM, True),
    )


@pytest.mark.parametrize("category", [Category.HUMAN, Category.MACHINE_UNAVAILABLE])
@pytest.mark.parametrize("previous", [Category.UNCERTAIN, Category.MACHINE_SCREENING])
def test_terminal_reply_policy(category: Category, previous: Category) -> None:
    state = send(request(stage(previous), 1), fsm.PredictionReceived(category), 1.1).state
    assert send(state, fsm.Signal.REPLY_REQUESTED).effects == (
        fsm.ReplyDecision(
            allow=category is Category.HUMAN,
            instructions_for=Category.HUMAN
            if category is Category.HUMAN and previous is Category.MACHINE_SCREENING
            else None,
        ),
    )
    assert send(state, fsm.Signal.REPLY_COMMITTED).effects == ()


def test_finish_is_idempotent_and_does_not_fabricate_predictions() -> None:
    result = send(request(ACTIVE), fsm.Finish(Reason.CANCELLED), 0.1)
    assert result.effects == (fsm.Action.CANCEL_CLASSIFICATION, fsm.Action.COMPLETE)
    assert result.state.work is None
    assert result.state.next_deadline is None
    for event in [
        fsm.Finish(Reason.INTERNAL_ERROR),
        fsm.PredictionReceived(Category.HUMAN),
        fsm.Signal.INFERENCE_FAILED,
        fsm.Signal.DEADLINE_REACHED,
    ]:
        assert send(result.state, event, 1) == fsm.Transition(result.state)
