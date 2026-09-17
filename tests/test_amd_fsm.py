from __future__ import annotations

import pytest

from livekit.agents.voice.amd import _fsm as fsm
from livekit.agents.voice.amd.events import AMDCategory as Category

pytestmark = pytest.mark.unit


def test_transition_is_repeatable_and_does_not_mutate_input() -> None:
    state = Category.UNCERTAIN
    event = Category.MACHINE_VM
    first = fsm.transition(state, event)
    assert fsm.transition(state, event) == first
    assert state is Category.UNCERTAIN
    assert first.next_state is Category.MACHINE_VM
    assert first.effects == ()


@pytest.mark.parametrize("current", list(Category))
@pytest.mark.parametrize("category", list(Category))
def test_stage_transitions(current: Category, category: Category) -> None:
    allowed = {
        Category.UNCERTAIN: set(Category),
        Category.MACHINE_SCREENING: {
            Category.MACHINE_SCREENING,
            Category.HUMAN,
            Category.MACHINE_VM,
            Category.MACHINE_UNAVAILABLE,
            Category.UNCERTAIN,
        },
        Category.MACHINE_VM: {
            Category.MACHINE_VM,
            Category.HUMAN,
            Category.MACHINE_IVR,
            Category.MACHINE_UNAVAILABLE,
            Category.UNCERTAIN,
        },
        Category.MACHINE_IVR: {
            Category.MACHINE_IVR,
            Category.HUMAN,
            Category.MACHINE_VM,
            Category.MACHINE_UNAVAILABLE,
            Category.UNCERTAIN,
        },
        Category.HUMAN: set(),
        Category.MACHINE_UNAVAILABLE: set(),
    }
    state = current
    event = category
    if category not in allowed[current]:
        with pytest.raises(ValueError, match="invalid AMD transition"):
            fsm.transition(state, event)
        return
    result = fsm.transition(state, event)
    assert result.next_state is category
    if category in {Category.HUMAN, Category.MACHINE_UNAVAILABLE}:
        assert result.effects == (fsm.Effect.COMPLETE,)
    elif category is Category.MACHINE_IVR:
        assert result.effects == (fsm.Effect.EXTRACT_MENU,)
    else:
        assert result.effects == ()


@pytest.mark.parametrize(
    ("initial", "corrected"),
    [
        (Category.MACHINE_SCREENING, Category.MACHINE_IVR),
        (Category.MACHINE_VM, Category.MACHINE_SCREENING),
        (Category.MACHINE_IVR, Category.MACHINE_SCREENING),
    ],
)
def test_uncertain_reopens_transitions(initial: Category, corrected: Category) -> None:
    state = initial
    with pytest.raises(ValueError, match="invalid AMD transition"):
        fsm.transition(state, corrected)
    uncertain = fsm.transition(state, Category.UNCERTAIN)
    assert uncertain.next_state is Category.UNCERTAIN
    assert uncertain.effects == ()
    result = fsm.transition(uncertain.next_state, corrected)
    assert result.next_state is corrected


def test_same_ivr_state_extracts_each_menu() -> None:
    state = Category.MACHINE_IVR
    result = fsm.transition(state, Category.MACHINE_IVR)
    assert result.next_state == state
    assert result.effects == (fsm.Effect.EXTRACT_MENU,)
