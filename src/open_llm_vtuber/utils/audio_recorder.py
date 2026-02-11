"""
Audio recorder for dual-channel (stereo) recording.

This module provides the AudioRecorder class for recording user input and TTS output
into a single stereo WAV file with user audio on channel 1 and TTS audio on channel 2.
"""

import os
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger


class AudioRecorder:
    """
    Records user input and TTS output into a dual-channel stereo audio file.

    The recorder maintains two separate buffers with timestamps:
    - Channel 1 (Left): User microphone input
    - Channel 2 (Right): TTS output (AI voice)

    Both channels are time-synchronized using timestamps to preserve conversation timing.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        """
        Initialize the audio recorder.

        Args:
            sample_rate: Sample rate for the audio recording in Hz (default: 16000)
        """
        self.sample_rate = sample_rate
        self._user_audio_segments: list[
            tuple[float, np.ndarray]
        ] = []  # (timestamp, audio)
        self._tts_audio_segments: list[
            tuple[float, np.ndarray]
        ] = []  # (timestamp, audio)
        self._start_time: Optional[float] = None
        self._tts_cumulative_duration: float = 0.0  # Track total TTS duration added
        self._tts_base_timestamp: Optional[float] = None  # When first TTS starts
        self._lock = asyncio.Lock()

        logger.debug(f"🎙️ AudioRecorder initialized with sample rate: {sample_rate} Hz")
        
    async def start(self) -> None:
        """
        Start the recording timer explicitly.
        
        This sets the reference start time for the recording. Any audio added later
        will be timestamped relative to this start time.
        """
        async with self._lock:
            if self._start_time is None:
                self._start_time = time.time()
                logger.info(f"🎙️ AudioRecorder started at {self._start_time}")

    async def add_user_audio(
        self, audio_data: np.ndarray, backdate_seconds: float = 0.0
    ) -> None:
        """
        Add user microphone audio to channel 1 with timestamp.

        Args:
            audio_data: Audio data as numpy array (float32)
            backdate_seconds: Number of seconds to backdate the timestamp
                              (useful when audio was captured earlier)
        """
        async with self._lock:
            if len(audio_data) > 0:
                current_time = 0.0
                
                # Initialize start time on first audio
                if self._start_time is None:
                    # If we receive a chunk, it represents audio that JUST finished.
                    # So the "start" of the conversation was `Now - Duration`.
                    duration = len(audio_data) / self.sample_rate
                    self._start_time = time.time() - duration - backdate_seconds
                    # First chunk always starts at 0.0 relative to this calculated start time
                    current_time = 0.0
                else:
                    # Subsequent chunks:
                    # We want to place them at (Now - Duration - Start_Time).
                    # This aligns the "Start" of this chunk with the timeline.
                    duration = len(audio_data) / self.sample_rate
                    # Timestamp should represent the START of this audio segment
                    current_time = time.time() - duration - self._start_time - backdate_seconds
                    # Ensure timestamp is never negative
                    current_time = max(0.0, current_time)

                self._user_audio_segments.append(
                    (current_time, audio_data.astype(np.float32))
                )

                logger.debug(
                    f"📝 Added {len(audio_data)} samples to user audio at {current_time:.2f}s "
                    f"(backdated: {backdate_seconds:.2f}s, total segments: {len(self._user_audio_segments)})"
                )

    async def add_tts_audio(self, audio_file_path: str) -> None:
        """
        Add TTS output audio to channel 2 with timestamp.

        TTS segments are placed consecutively based on cumulative duration,
        but will sync with wall-clock time if there's a significant gap
        (indicating a new turn).

        Args:
            audio_file_path: Path to the TTS audio file
        """
        async with self._lock:
            try:
                # Initialize start time on first audio
                if self._start_time is None:
                    self._start_time = time.time()

                # Read the audio file
                audio_data, file_sample_rate = sf.read(audio_file_path, dtype="float32")

                # Convert stereo to mono if needed
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=1)

                # Resample if sample rate doesn't match
                if file_sample_rate != self.sample_rate:
                    audio_data = self._resample(
                        audio_data, file_sample_rate, self.sample_rate
                    )

                current_wall_time = time.time() - self._start_time
                segment_duration = len(audio_data) / self.sample_rate

                # Logic to determine timestamp:
                # 1. If this is the FIRST TTS segment ever -> use current wall time
                # 2. If there is a significant gap (> 0.5s) between where the last TTS ended
                #    and now -> assume new turn, use current wall time
                # 3. Otherwise -> use cumulative duration (continue stream)

                predicted_tts_end = 0.0
                if self._tts_base_timestamp is not None:
                    predicted_tts_end = (
                        self._tts_base_timestamp + self._tts_cumulative_duration
                    )

                # Check gap: Current time vs Predicted end of previous TTS
                # If gap > 0.5s, we sync to wall clock (Scenario 1: Pauses)
                if (
                    self._tts_base_timestamp is None
                    or (current_wall_time - predicted_tts_end) > 0.5
                ):
                    self._tts_base_timestamp = current_wall_time
                    self._tts_cumulative_duration = 0.0
                    timestamp = self._tts_base_timestamp
                else:
                    timestamp = self._tts_base_timestamp + self._tts_cumulative_duration

                # Append to TTS segments
                self._tts_audio_segments.append(
                    (timestamp, audio_data.astype(np.float32))
                )

                # Update cumulative duration
                self._tts_cumulative_duration += segment_duration

                logger.debug(
                    f"📝 Added TTS audio from {audio_file_path} at {timestamp:.2f}s "
                    f"(duration: {segment_duration:.2f}s, base: {self._tts_base_timestamp:.2f}s, "
                    f"stream_pos: {self._tts_cumulative_duration:.2f}s)"
                )

            except Exception as e:
                logger.error(f"❌ Error loading TTS audio file {audio_file_path}: {e}")

    async def handle_interruption(self, timestamp: float) -> None:
        """
        Handle interruption by truncating future TTS audio.

        When user interrupts (Scenario 2), we should stop 'recording' TTS
        immediately. This means removing/truncating any TTS segments that
        go beyond the interruption timestamp.

        Args:
            timestamp: The unix timestamp (time.time()) when interruption occurred
        """
        async with self._lock:
            if self._start_time is None:
                return

            # Calculate relative timestamp
            interrupt_rel_time = timestamp - self._start_time
            if interrupt_rel_time < 0:
                interrupt_rel_time = 0.0

            logger.info(
                f"🛑 Handling interruption at {interrupt_rel_time:.2f}s (wall time: {timestamp})"
            )

            new_segments = []
            for tts_start, audio in self._tts_audio_segments:
                tts_end = tts_start + (len(audio) / self.sample_rate)

                if tts_start >= interrupt_rel_time:
                    # Segment is entirely in the future -> Remove it
                    continue
                elif tts_end > interrupt_rel_time:
                    # Segment overlaps -> Truncate it
                    cutoff_duration = interrupt_rel_time - tts_start
                    cutoff_samples = int(cutoff_duration * self.sample_rate)
                    truncated_audio = audio[:cutoff_samples]
                    new_segments.append((tts_start, truncated_audio))
                else:
                    # Segment is entirely in the past -> Keep it
                    new_segments.append((tts_start, audio))

            self._tts_audio_segments = new_segments

            # Reset cumulative logic so next TTS starts fresh (syncs to wall clock)
            self._tts_base_timestamp = None
            self._tts_cumulative_duration = 0.0

    def _resample(
        self, audio_data: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """
        Resample audio data to target sample rate using linear interpolation.

        Args:
            audio_data: Input audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled audio data
        """
        if orig_sr == target_sr:
            return audio_data

        # Calculate the ratio and new length
        duration = len(audio_data) / orig_sr
        new_length = int(duration * target_sr)

        # Create interpolation indices
        old_indices = np.linspace(0, len(audio_data) - 1, len(audio_data))
        new_indices = np.linspace(0, len(audio_data) - 1, new_length)

        # Interpolate
        resampled = np.interp(new_indices, old_indices, audio_data)

        logger.debug(
            f"🔄 Resampled audio from {orig_sr} Hz to {target_sr} Hz "
            f"({len(audio_data)} → {len(resampled)} samples)"
        )

        return resampled.astype(np.float32)

    def _build_timeline(
        self, segments: list[tuple[float, np.ndarray]], total_duration: float
    ) -> np.ndarray:
        """
        Build a complete audio timeline with proper timing from segments.

        Args:
            segments: List of (timestamp, audio_data) tuples
            total_duration: Total duration of the recording in seconds

        Returns:
            Complete audio array with silence padding
        """
        # Calculate total samples needed
        total_samples = int(total_duration * self.sample_rate)
        timeline = np.zeros(total_samples, dtype=np.float32)

        for timestamp, audio_data in segments:
            # Calculate starting position in samples
            start_sample = int(timestamp * self.sample_rate)
            end_sample = start_sample + len(audio_data)

            # Ensure we don't exceed array bounds
            if end_sample > total_samples:
                end_sample = total_samples
                audio_data = audio_data[: end_sample - start_sample]

            # Place audio at the correct timestamp
            timeline[start_sample:end_sample] = audio_data

        return timeline

    async def save_recording(self, output_path: str) -> bool:
        """
        Save the dual-channel recording to a WAV file with proper timing.

        The user audio will be on channel 1 (left) and TTS audio on channel 2 (right).
        Timing is preserved using timestamps so pauses and turn-taking are maintained.

        Args:
            output_path: Path where the WAV file will be saved

        Returns:
            True if save was successful, False otherwise
        """
        async with self._lock:
            try:
                # Check if we have any audio to save
                if (
                    len(self._user_audio_segments) == 0
                    and len(self._tts_audio_segments) == 0
                ):
                    logger.warning("⚠️ No audio data to save")
                    return False

                # Calculate total duration from all segments
                max_user_time = 0.0
                max_tts_time = 0.0

                if self._user_audio_segments:
                    last_timestamp, last_audio = self._user_audio_segments[-1]
                    max_user_time = last_timestamp + (
                        len(last_audio) / self.sample_rate
                    )

                if self._tts_audio_segments:
                    last_timestamp, last_audio = self._tts_audio_segments[-1]
                    max_tts_time = last_timestamp + (len(last_audio) / self.sample_rate)

                total_duration = max(max_user_time, max_tts_time)

                # Build timelines for both channels
                user_channel = self._build_timeline(
                    self._user_audio_segments, total_duration
                )
                tts_channel = self._build_timeline(
                    self._tts_audio_segments, total_duration
                )

                # Create stereo audio by stacking channels
                stereo_audio = np.stack([user_channel, tts_channel], axis=1)

                # Ensure output directory exists
                output_dir = Path(output_path).parent
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save as WAV file
                sf.write(output_path, stereo_audio, self.sample_rate, subtype="PCM_16")

                logger.info(
                    f"💾 Recording saved to {output_path} "
                    f"(duration: {total_duration:.2f}s, channels: 2, "
                    f"user segments: {len(self._user_audio_segments)}, "
                    f"tts segments: {len(self._tts_audio_segments)})"
                )

                return True

            except Exception as e:
                logger.error(f"❌ Error saving recording to {output_path}: {e}")
                return False

    async def clear(self) -> None:
        """
        Clear all audio segments and reset the recorder.

        This should be called when starting a new recording session.
        """
        async with self._lock:
            user_segments = len(self._user_audio_segments)
            tts_segments = len(self._tts_audio_segments)

            self._user_audio_segments = []
            self._tts_audio_segments = []
            self._start_time = None
            self._tts_cumulative_duration = 0.0
            self._tts_base_timestamp = None

            logger.debug(
                f"🧹 Cleared audio segments (user: {user_segments}, tts: {tts_segments})"
            )

    def has_audio(self) -> bool:
        """
        Check if the recorder has any audio data.

        Returns:
            True if there is audio data in either channel, False otherwise
        """
        return len(self._user_audio_segments) > 0 or len(self._tts_audio_segments) > 0

    def get_duration(self) -> float:
        """
        Get the current duration of the recording in seconds.

        Returns:
            Duration in seconds based on the longest channel
        """
        if self._start_time is None or not self.has_audio():
            return 0.0

        max_duration = 0.0

        if self._user_audio_segments:
            last_timestamp, last_audio = self._user_audio_segments[-1]
            max_duration = max(
                max_duration, last_timestamp + (len(last_audio) / self.sample_rate)
            )

        if self._tts_audio_segments:
            last_timestamp, last_audio = self._tts_audio_segments[-1]
            max_duration = max(
                max_duration, last_timestamp + (len(last_audio) / self.sample_rate)
            )

        return max_duration

    @staticmethod
    def generate_filename(
        conf_uid: str, history_uid: Optional[str] = None, output_dir: str = "recordings"
    ) -> str:
        """
        Generate a filename for the recording.

        Args:
            conf_uid: Configuration UID
            history_uid: Chat history UID (optional)
            output_dir: Output directory for recordings

        Returns:
            Full path to the recording file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if history_uid:
            filename = f"recording_{conf_uid}_{history_uid}_{timestamp}.wav"
        else:
            filename = f"recording_{conf_uid}_{timestamp}.wav"

        return os.path.join(output_dir, filename)
