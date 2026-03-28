"""Temporal memory: observations and how they read over time.

Three layers (this module cares most about the first two):

1. **Observations** — raw per-frame signals (“arms raised”, “crouching”).
2. **Temporal summary** — compact *timeline* of those signals over time. True
   “what it means” pattern text (e.g. sports-like repetitive overhead motion)
   usually comes from a separate summarizer / LLM that reads observations +
   ``temporal_summary`` as context — not from storing old guesses.
3. **Guess evolution** — optional belief smoothing (candidates, anti-flip). Kept
   here as thin glue for one-shot pipelines; strategy should lean on (1)+(2).

Good guesses follow good temporal understanding, not from idolizing past labels.
"""
from __future__ import annotations

from copy import deepcopy
from typing import List

from agent.nodes.state import AgentState, FrameObservation, CandidateGuess


# -----------------------------
# CONFIG (tune fast during demo)
# -----------------------------
MAX_OBSERVATIONS = 5
CONFIDENCE_THRESHOLD = 0.1
FINAL_CONFIDENCE = 0.85


def _collapse_consecutive(strings: List[str]) -> List[str]:
    out: List[str] = []
    for s in strings:
        t = s.strip()
        if not t:
            continue
        if not out or out[-1] != t:
            out.append(t)
    return out


# -----------------------------
# LAYER 1 — OBSERVATIONS (what we saw)
# -----------------------------
def add_observation(
    state: AgentState,
    summary: str,
    raw_response: str,
    timestamp: float,
) -> AgentState:
    new_state = deepcopy(state)

    observation: FrameObservation = {
        "timestamp": timestamp,
        "summary": summary,
        "raw_response": raw_response,
    }

    new_state["observations"].append(observation)

    if len(new_state["observations"]) > MAX_OBSERVATIONS:
        new_state["observations"] = new_state["observations"][-MAX_OBSERVATIONS:]

    return new_state


# -----------------------------
# LAYER 2 — TEMPORAL LINE (for pattern / meaning — feed to summarizer or LLM)
# -----------------------------
def build_temporal_summary(state: AgentState) -> str:
    """Compact timeline of observation summaries, older → newer.

    Consecutive duplicates collapse so "jump" repeated three frames becomes one step.
    High-level interpretation belongs in a downstream node that reads
    ``state['observations']`` (incl. raw_response) plus this string.
    """
    observations = state["observations"]
    if not observations:
        return ""

    summaries = [obs["summary"] for obs in observations]
    summaries = _collapse_consecutive(summaries)
    if not summaries:
        return ""
    if len(summaries) == 1:
        return summaries[0]
    return " → ".join(summaries)


def update_temporal_summary(state: AgentState) -> AgentState:
    new_state = deepcopy(state)
    new_state["temporal_summary"] = build_temporal_summary(state)
    return new_state


def ingest_observations_and_summary(
    state: AgentState,
    summary: str,
    raw_response: str,
    timestamp: float,
) -> AgentState:
    """Layers 1–2 only: append observation and refresh ``temporal_summary``."""
    state = add_observation(state, summary, raw_response, timestamp)
    return update_temporal_summary(state)


# -----------------------------
# LAYER 3 — BELIEF (optional; thin smoothing, not the core of this node)
# -----------------------------
def update_candidates(
    state: AgentState,
    new_candidates: List[CandidateGuess],
) -> AgentState:
    new_state = deepcopy(state)

    if not new_candidates:
        return new_state

    best = max(new_candidates, key=lambda x: x["confidence"])

    prev_best = new_state.get("best_guess")
    prev_conf = new_state.get("best_confidence", 0)

    new_best = best["label"]
    new_conf = best["confidence"]

    should_update = False
    if prev_best is None:
        should_update = True
    elif new_best == prev_best:
        should_update = True
    elif new_conf > prev_conf + CONFIDENCE_THRESHOLD:
        should_update = True

    if should_update:
        new_state["candidates"] = new_candidates
        new_state["best_guess"] = new_best
        new_state["best_confidence"] = new_conf

    return new_state


def check_should_finalize(state: AgentState) -> AgentState:
    new_state = deepcopy(state)
    best_conf = new_state.get("best_confidence", 0)
    new_state["should_finalize"] = best_conf >= FINAL_CONFIDENCE
    return new_state


# -----------------------------
# Full stack (temporal core + optional belief)
# -----------------------------
def update_temporal_memory(
    state: AgentState,
    summary: str,
    raw_response: str,
    timestamp: float,
    new_candidates: List[CandidateGuess],
) -> AgentState:
    """Run layer 1–2 first, then optional candidate smoothing + finalize flag."""
    state = ingest_observations_and_summary(state, summary, raw_response, timestamp)
    state = update_candidates(state, new_candidates)
    state = check_should_finalize(state)
    return state
