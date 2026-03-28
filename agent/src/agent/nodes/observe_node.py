"""Sample observer helpers that use agent-side video sampling.

These functions are examples of how to connect a frame source to
VideoSamplingService without modifying `api/` or `core/`.
"""

from __future__ import annotations

from core import Frame
from typing import AsyncIterator

from core import start_practice, start_stream
from prompts.observe_prompt import OBSERVE_PROMPT
from services.llm import OpenRouterClient, try_parse_json
from services.video import BufferedSample, sample_video_feed


async def observe_frame(frame: Frame) -> list[str]:
	"""Call OpenRouter vision model and return structured observations."""
	client = OpenRouterClient()
	raw = await client.run_vision(
		system_prompt=OBSERVE_PROMPT,
		user_prompt=(
			"Observe this single frame and return valid JSON with exactly 3 short "
			"literal observations in the `observations` array."
		),
		image=frame.image,
	)

	try:
		parsed = try_parse_json(raw)
	except Exception:
		# Fallback: preserve model output in a single observation if JSON parse fails.
		return [raw.strip()]

	observations = parsed.get("observations", [])
	if not isinstance(observations, list):
		return [raw.strip()]

	out: list[str] = []
	for item in observations:
		if isinstance(item, str) and item.strip():
			out.append(item.strip())

	return out[:3] if out else [raw.strip()]


async def observe_practice_video(
	camera_index: int = 0,
	input_fps: int = 8,
	sample_fps: float = 3.0,
	buffer_seconds: float = 2.5,
) -> AsyncIterator[BufferedSample]:
	"""Observe local camera feed with sampling and 2-3s buffering.

	Args:
		camera_index: Camera device index.
		input_fps: Raw capture rate from practice mode.
		sample_fps: Output sampling FPS (2-5).
		buffer_seconds: Rolling context window in seconds (2-3).
	"""
	frames = start_practice(camera_index=camera_index, fps=input_fps)
	async for sample in sample_video_feed(
		frames,
		target_fps=sample_fps,
		buffer_seconds=buffer_seconds,
	):
		yield sample


async def observe_live_video(
	livekit_url: str,
	token: str,
	sample_fps: float = 2.0,
	buffer_seconds: float = 2.5,
) -> AsyncIterator[BufferedSample]:
	"""Observe live stream with sampling and buffering.

	Note:
		`core.start_stream()` currently emits at roughly 1 FPS upstream.
		Sampling settings here still apply but cannot increase source FPS.
	"""
	frames = start_stream(livekit_url, token)
	async for sample in sample_video_feed(
		frames,
		target_fps=sample_fps,
		buffer_seconds=buffer_seconds,
	):
		yield sample
