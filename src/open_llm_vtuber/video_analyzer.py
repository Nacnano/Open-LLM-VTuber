"""Video analysis module for conversation feedback using Visual LLMs.

This module extracts frames from recorded meeting videos, combines them with
the chat transcript, and sends them to a Visual LLM (OpenAI-compatible API
with vision support) to generate conversational skill feedback.
"""

import asyncio
import base64
import subprocess
from pathlib import Path

# from openai import AsyncOpenAI
from google import genai
from loguru import logger

from .video_analyzer_prompts import DEFAULT_ANALYSIS_PROMPT

def _extract_frames(video_path: str, max_frames: int = 8) -> list[str]:
    """Extract evenly-spaced frames from a video file as base64-encoded JPEGs.

    Uses ffprobe to get duration and ffmpeg to extract frames at regular intervals.

    Args:
        video_path: Path to the video file.
        max_frames: Maximum number of frames to extract.

    Returns:
        List of base64-encoded JPEG strings.
    """
    # Get video duration
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        duration = float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        logger.error(f"Failed to get video duration: {e}")
        return []

    if duration <= 0:
        logger.warning(f"Video duration is {duration}s, skipping frame extraction")
        return []

    # Calculate timestamps for evenly spaced frames
    interval = duration / (max_frames + 1)
    timestamps = [interval * (i + 1) for i in range(max_frames)]

    frames: list[str] = []
    for ts in timestamps:
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-ss",
                    str(ts),
                    "-i",
                    video_path,
                    "-vframes",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "mjpeg",
                    "-q:v",
                    "5",
                    "-vf",
                    "scale=640:-1",
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            if result.stdout:
                b64 = base64.b64encode(result.stdout).decode("utf-8")
                frames.append(b64)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to extract frame at {ts:.1f}s: {e}")
            continue

    logger.info(f"📸 Extracted {len(frames)} frames from video ({duration:.1f}s)")
    return frames


async def extract_frames_async(video_path: str, max_frames: int = 8) -> list[str]:
    """Extract frames from a video file asynchronously.

    Runs the blocking ffmpeg operations in a thread pool to avoid blocking
    the event loop.

    Args:
        video_path: Path to the video file.
        max_frames: Maximum number of frames to extract.

    Returns:
        List of base64-encoded JPEG strings.
    """
    return await asyncio.to_thread(_extract_frames, video_path, max_frames)


async def analyze_video(
    video_path: str,
    transcript: str,
    base_url: str,
    api_key: str,
    model: str,
    analysis_prompt: str | None = None,
    max_frames: int = 8,
    temperature: float = 0.7,
) -> str:
    """Analyze a recorded conversation video using a Visual LLM.

    Extracts frames from the video, combines them with the conversation
    transcript, and sends to a vision-capable LLM for analysis.

    Args:
        video_path: Path to the recorded video file.
        transcript: Text transcript of the conversation.
        base_url: Base URL for the OpenAI-compatible API.
        api_key: API key for authentication.
        model: Model name to use (must support vision).
        analysis_prompt: Custom system prompt for analysis. Uses default if None.
        max_frames: Maximum number of frames to extract from video.
        temperature: Sampling temperature for the LLM.

    Returns:
        The analysis feedback text from the LLM.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If frame extraction or LLM call fails.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Read video file as bytes
    video_bytes = await asyncio.to_thread(Path(video_path).read_bytes)
    logger.info(f"📹 Read video file: {len(video_bytes)} bytes")

    # Build the prompt
    system_prompt = analysis_prompt or DEFAULT_ANALYSIS_PROMPT
    client = genai.Client(vertexai=True)

    # Prepare video part with mime type
    video_part = genai.types.Part.from_bytes(
        data=video_bytes,
        mime_type="video/mp4"  # Adjust if your videos use different format
    )
    logger.info("📦 Prepared video part for Vertex AI")

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=[video_part, system_prompt],
        )

        feedback = response.text


        # feedback = (
        #     "This is a placeholder\n"
        #     f"Using video path: {video_path}\n"
        #     f" Transcript length: {len(transcript)} characters.\n\n"
        #     f"Transcript\n{transcript}..."
        # )
        logger.info("✅ Video analysis complete")
        return feedback

    except Exception as e:
        logger.error(f"❌ Visual LLM analysis failed: {e}")
        raise RuntimeError(f"Video analysis failed: {e}") from e
