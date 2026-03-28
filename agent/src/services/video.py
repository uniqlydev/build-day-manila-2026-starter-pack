"""Video sampling utilities for agent-side frame processing.

This module is intentionally self-contained under `agent/` so it can be
customized without touching `api/` or `core/`.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import platform
import shutil
from typing import AsyncIterator, Deque, List

from PIL import Image

from core import Frame


@dataclass(frozen=True)
class BufferedSample:
	"""A sampled frame plus the rolling buffered context up to that frame."""

	frame: Frame
	buffered_frames: List[Frame]


class VideoSamplingService:
	"""Capture frames, sample at a fixed FPS, and keep a short rolling buffer.

	Constraints:
	- target_fps: 2.0 to 5.0
	- buffer_seconds: 2.0 to 3.0
	"""

	def __init__(self, target_fps: float = 2.0, buffer_seconds: float = 2.0) -> None:
		if not 2.0 <= target_fps <= 5.0:
			raise ValueError("target_fps must be between 2 and 5")
		if not 2.0 <= buffer_seconds <= 3.0:
			raise ValueError("buffer_seconds must be between 2 and 3")

		self.target_fps = target_fps
		self.buffer_seconds = buffer_seconds
		self._sample_interval = timedelta(seconds=1.0 / target_fps)
		self._window = timedelta(seconds=buffer_seconds)

		self._buffer: Deque[Frame] = deque()
		self._last_sample_time: datetime | None = None

	def reset(self) -> None:
		"""Clear internal sampling and buffering state."""
		self._buffer.clear()
		self._last_sample_time = None

	def _prune_buffer(self, now: datetime) -> None:
		cutoff = now - self._window
		while self._buffer and self._buffer[0].timestamp < cutoff:
			self._buffer.popleft()

	def capture(self, frame: Frame) -> bool:
		"""Capture a frame if it satisfies the current FPS sampling interval.

		Returns:
			True if the frame is accepted as a sampled frame, else False.
		"""
		if self._last_sample_time is not None:
			if frame.timestamp - self._last_sample_time < self._sample_interval:
				return False

		self._last_sample_time = frame.timestamp
		self._buffer.append(frame)
		self._prune_buffer(frame.timestamp)
		return True

	def get_buffered_frames(self) -> List[Frame]:
		"""Return buffered sampled frames (new list, oldest to newest)."""
		return list(self._buffer)

	def get_latest(self) -> Frame | None:
		"""Return the latest sampled frame from the buffer if available."""
		return self._buffer[-1] if self._buffer else None

	async def stream_samples(
		self,
		frames: AsyncIterator[Frame],
	) -> AsyncIterator[BufferedSample]:
		"""Yield sampled frames with their rolling 2-3s context window.

		Example:
			service = VideoSamplingService(target_fps=3.0, buffer_seconds=2.5)
			async for sample in service.stream_samples(start_practice(...)):
				newest = sample.frame
				context = sample.buffered_frames
		"""
		async for frame in frames:
			if not self.capture(frame):
				continue

			yield BufferedSample(
				frame=frame,
				buffered_frames=self.get_buffered_frames(),
			)


async def sample_video_feed(
	frames: AsyncIterator[Frame],
	target_fps: float = 2.0,
	buffer_seconds: float = 2.0,
) -> AsyncIterator[BufferedSample]:
	"""Convenience wrapper around VideoSamplingService.

	This is useful if you prefer a function-style API.
	"""
	service = VideoSamplingService(
		target_fps=target_fps,
		buffer_seconds=buffer_seconds,
	)

	async for sample in service.stream_samples(frames):
		yield sample


def _detect_ffmpeg() -> str:
	path = shutil.which("ffmpeg")
	if path:
		return path

	try:
		import imageio_ffmpeg

		return imageio_ffmpeg.get_ffmpeg_exe()
	except Exception:
		pass

	raise FileNotFoundError(
		"ffmpeg not found. Install it:\n"
		"  Linux:  sudo apt install ffmpeg\n"
		"  macOS:  brew install ffmpeg\n"
		"  Windows: winget install ffmpeg"
	)


def _build_capture_cmd(
	ffmpeg: str,
	camera_index: int,
	video_size: str | None = None,
) -> list[str]:
	system = platform.system()

	if system == "Linux":
		input_fmt = ["-f", "v4l2"]
		device = f"/dev/video{camera_index}"
	elif system == "Darwin":
		input_fmt = ["-f", "avfoundation", "-framerate", "30"]
		device = str(camera_index)
	elif system == "Windows":
		input_fmt = ["-f", "dshow"]
		device = f"video={camera_index}"
	else:
		input_fmt = ["-f", "v4l2"]
		device = f"/dev/video{camera_index}"

	size_args = ["-video_size", video_size] if video_size else []

	return [
		ffmpeg,
		"-hide_banner",
		"-loglevel",
		"error",
		*input_fmt,
		*size_args,
		"-i",
		device,
		"-vframes",
		"1",
		"-f",
		"rawvideo",
		"-pix_fmt",
		"rgb24",
		"-vcodec",
		"rawvideo",
		"pipe:1",
	]


def _guess_dimensions(num_bytes: int) -> tuple[int, int] | None:
	if num_bytes <= 0 or num_bytes % 3 != 0:
		return None

	pixels = num_bytes // 3

	common_dims = [
		(3840, 2160),
		(2560, 1440),
		(2048, 1536),
		(1920, 1440),
		(1920, 1080),
		(1600, 1200),
		(1440, 1080),
		(1280, 960),
		(1280, 720),
		(1024, 768),
		(800, 600),
		(640, 480),
		(320, 240),
	]

	for width, height in common_dims:
		if width * height == pixels:
			return width, height

	candidate_widths = [
		3840,
		2560,
		2048,
		1920,
		1680,
		1600,
		1440,
		1366,
		1280,
		1024,
		960,
		854,
		800,
		768,
		720,
		640,
		480,
		426,
		320,
	]

	for width in candidate_widths:
		if pixels % width != 0:
			continue
		height = pixels // width
		if 120 <= height <= 4320:
			return width, height

	return None


async def _capture_one_frame(cmd: list[str]) -> Image.Image:
	proc = await asyncio.create_subprocess_exec(
		*cmd,
		stdout=asyncio.subprocess.PIPE,
		stderr=asyncio.subprocess.PIPE,
	)
	stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)

	if proc.returncode != 0:
		err = stderr.decode(errors="replace").strip()
		raise RuntimeError(f"ffmpeg capture failed (exit {proc.returncode}): {err}")

	if not stdout:
		raise RuntimeError("ffmpeg returned no data")

	dims = _guess_dimensions(len(stdout))
	if dims is None:
		raise RuntimeError(
			f"Could not determine frame dimensions from {len(stdout)} bytes of raw data. "
			"Try setting --video-size (for example: 1920x1440, 1280x720, or 640x480)."
		)

	return Image.frombytes("RGB", dims, stdout)


async def start_practice_frames(
	camera_index: int = 0,
	fps: int = 1,
	video_size: str | None = None,
) -> AsyncIterator[Frame]:
	"""Yield frames from a local camera without relying on core practice capture.

	Args:
		camera_index: Camera device index.
		fps: Capture rate (frames per second).
		video_size: Optional size override (e.g., "1280x720").
	"""
	if fps <= 0:
		raise ValueError("fps must be greater than 0")

	interval = 1.0 / fps

	print(f"[practice] Opening camera {camera_index}...")
	print(f"[practice] Sampling at {fps} FPS. Press Ctrl+C to stop.\n")

	try:
		ffmpeg = _detect_ffmpeg()
	except FileNotFoundError as exc:
		print(f"[!] {exc}")
		return

	cmd = _build_capture_cmd(ffmpeg, camera_index, video_size=video_size)

	try:
		test_frame = await _capture_one_frame(cmd)
		print(
			f"[practice] Camera {camera_index} ready "
			f"({test_frame.size[0]}x{test_frame.size[1]}).\n"
		)
	except Exception as exc:
		print(f"[!] Could not capture from camera {camera_index}: {exc}")
		return

	while True:
		try:
			image = await _capture_one_frame(cmd)

			yield Frame(
				image=image,
				timestamp=datetime.now(timezone.utc),
			)

			await asyncio.sleep(interval)

		except KeyboardInterrupt:
			print("\n[practice] Stopped.")
			break
		except Exception as exc:
			print(f"[practice] Error capturing frame: {exc}")
			break
