"""Turn identity, classifier context, and speech timing owned by the AMD coordinator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from .events import AMDCategory, AMDPredictionEvent

AMDTranscriptSource = Literal["session", "amd"]
_HISTORY_LIMIT = 19


@dataclass(frozen=True)
class AMDTranscript:
    transcript: str
    source: AMDTranscriptSource | None


class AMDTurnContext(BaseModel):
    turn_id: int
    transcript: str
    transcript_source: AMDTranscriptSource | None
    dtmf_digits: str


class AMDClassifyRequest(BaseModel):
    """Model-facing payload for one classification. Serialize with ``exclude_none``."""

    stage: AMDCategory
    allowed_next_categories: list[AMDCategory]
    current_turn: AMDTurnContext
    earlier_turns: list[AMDTurnContext]
    speech_duration: float


@dataclass
class PredictionSlot:
    result: AMDPredictionEvent | None = None


@dataclass
class Turn:
    turn_id: int
    committed_at: float
    transcript: AMDTranscript
    speech_duration: float
    dtmf_digits: str
    prediction: PredictionSlot = field(default_factory=PredictionSlot)
    inference_duration: float | None = None

    def context(self) -> AMDTurnContext:
        return AMDTurnContext(
            turn_id=self.turn_id,
            transcript=self.transcript.transcript,
            transcript_source=self.transcript.source,
            dtmf_digits=self.dtmf_digits,
        )


@dataclass
class SpeechWindow:
    """User speech edges on the monotonic clock. The speech window resets at each commit."""

    speaking_since: float | None = None
    silence_since: float | None = None
    uncommitted: bool = False
    """Speech started after the last commit, so no turn has claimed it yet."""
    duration: float = 0.0
    """Speech time accumulated for the next commit."""

    @property
    def speaking(self) -> bool:
        return self.speaking_since is not None

    def started(self, at: float) -> None:
        self.speaking_since = at
        self.silence_since = None
        self.uncommitted = True

    def ended(self, at: float) -> None:
        if self.speaking_since is not None:
            self.duration += max(0.0, at - self.speaking_since)
        self.speaking_since = None
        self.silence_since = at

    def commit(self, now: float, eot_delay: float) -> float:
        """Close the turn's speech window and return its speech duration."""
        if self.speaking_since is None and not self.uncommitted:
            # No speech edge since the last commit: the EOT delay is the freshest anchor.
            self.silence_since = now - max(0.0, eot_delay)
        self.uncommitted = False
        duration, self.duration = self.duration, 0.0
        if self.speaking_since is not None:
            duration += now - self.speaking_since
            self.speaking_since = now
        return duration


class Turns:
    def __init__(self) -> None:
        self._turns: dict[int, Turn] = {}
        self._pending_dtmf_digits = ""

    def __contains__(self, turn_id: object) -> bool:
        return turn_id in self._turns

    @property
    def turn_id(self) -> int:
        return next(reversed(self._turns), 0)

    def prediction(self, turn_id: int) -> AMDPredictionEvent | None:
        return self._turns[turn_id].prediction.result

    def dtmf_sent(self, digits: str) -> None:
        if not digits or any(digit not in "0123456789*#ABCD" for digit in digits):
            raise ValueError("digits must contain only 0-9, *, #, A-D")
        self._pending_dtmf_digits += digits

    def clear_pending_dtmf(self) -> None:
        self._pending_dtmf_digits = ""

    def commit(self, transcript: AMDTranscript, speech_duration: float, now: float) -> Turn:
        turn = Turn(self.turn_id + 1, now, transcript, speech_duration, self._pending_dtmf_digits)
        self._pending_dtmf_digits = ""
        self._turns[turn.turn_id] = turn
        return turn

    def request(
        self, turn: Turn, *, stage: AMDCategory, allowed: list[AMDCategory]
    ) -> AMDClassifyRequest:
        earlier = [t for t in self._turns.values() if t.turn_id < turn.turn_id]
        return AMDClassifyRequest(
            stage=stage,
            allowed_next_categories=allowed,
            current_turn=turn.context(),
            earlier_turns=[t.context() for t in earlier[-_HISTORY_LIMIT:]],
            speech_duration=turn.speech_duration,
        )
