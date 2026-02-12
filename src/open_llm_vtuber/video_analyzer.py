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

# Default analysis prompt for IELTS-style conversation evaluation
DEFAULT_ANALYSIS_PROMPT = """You are an expert conversation coach and IELTS examiner. 
Analyze the following conversation recording between a student and an AI examiner.

You are given:
1. Extracted video frames showing the student during the conversation
2. The full text transcript of the conversation

Please provide detailed feedback on the student's conversational performance, including:

## Overall Score (out of 9, IELTS band scale)

## Fluency & Coherence
- How smoothly did the student speak?
- Were there noticeable pauses, hesitations, or repetitions?
- Did the student organize their ideas logically?

## Lexical Resource (Vocabulary)
- Range and accuracy of vocabulary used
- Use of idiomatic expressions or collocations
- Any vocabulary errors or limitations

## Grammatical Range & Accuracy
- Variety of sentence structures used
- Accuracy of grammar
- Common grammatical errors noticed

## Pronunciation & Delivery (based on video frames)
- Body language and facial expressions
- Eye contact and engagement
- Overall confidence and composure

## Key Strengths
- List 2-3 specific things the student did well

## Areas for Improvement
- List 2-3 specific areas to work on with actionable suggestions

## Sample Improved Responses
- Pick 1-2 of the student's weaker responses and provide improved versions

Keep your feedback constructive, specific, and encouraging."""


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

    # Extract frames

    # Build the prompt
    system_prompt = analysis_prompt or DEFAULT_ANALYSIS_PROMPT
    client = genai.Client(api_key=api_key)
    file  = await client.aio.files.upload(file=video_path)

    try:
        # response = await client.chat.completions.create(
        #     model=model,
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": content},
        #     ],
        #     temperature=temperature,
        #     max_tokens=4096,
        # )

        # feedback = response.choices[0].message.content

        response = await client.aio.models.generate_content(
            model=model,
            contents=[file, system_prompt],
        )

        feedback = response.text


        # feedback = (
        #     "This is a placeholder\n"
        #     f"Converation analysis for video {video_path} with {len(frames)} frames.\n\n"
        #     f"Using video path: {video_path}\n"
        #     f" Transcript length: {len(transcript)} characters.\n\n"
        #     f"Transcript\n{transcript}..."
        # )
        logger.info("✅ Video analysis complete")
        return feedback

    except Exception as e:
        logger.error(f"❌ Visual LLM analysis failed: {e}")
        raise RuntimeError(f"Video analysis failed: {e}") from e
