from typing import TypedDict, List, Dict, Optional

class FrameObservation(TypedDict):
    timestamp: float
    summary: str
    raw_response: str

class CandidateGuess(TypedDict):
    label: str
    confidence: float
    reason: str

class AgentState(TypedDict):
    # latest input
    current_frame_path: Optional[str]

    # rolling temporal memory
    observations: List[FrameObservation]

    # compressed temporal understanding
    temporal_summary: str

    # current model outputs
    candidates: List[CandidateGuess]
    best_guess: Optional[str]
    best_confidence: float

    # optional control flags
    should_finalize: bool
