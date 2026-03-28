"""System prompt and analysis logic for the guessing game agent.

=== EDIT THIS FILE ===

This is where you define your agent's strategy:
- What system prompt to use
- How to analyze each frame
- When to submit a guess vs. gather more context
"""

from __future__ import annotations

from agent.nodes.observe_node import observe_frame
from core import Frame
from prompts.observe_prompt import OBSERVE_PROMPT

# ---------------------------------------------------------------------------
# System prompt — tweak this to improve your agent's guessing ability.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = OBSERVE_PROMPT


async def analyze(frame: Frame) -> str | None:
    """Analyze a single frame and return a guess, or None to skip.

    This is the core function you should customize. The default
    implementation is a simple placeholder that always skips.

    Args:
        frame: A Frame with .image (PIL Image) and .timestamp.

    Returns:
        A text guess string, or None to skip this frame.
    """
    observations = await observe_frame(frame)
    print(f"  [observe] {observations}")

    # Observation-only mode: do not submit a gameplay guess yet.
    return None
