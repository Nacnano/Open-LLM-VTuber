"""
Audio recorder for dual-channel (stereo) recording.

This module provides the AudioRecorder class for recording user input and TTS output
into a single stereo WAV file with user audio on channel 1 and TTS audio on channel 2.

The recorder uses a hybrid wall-clock / cursor model for turn-based conversations:
- Wall-clock timestamps are used so that silence between turns is preserved
  (keeping audio in sync with the video recording).
- A cursor tracks the end of the last placed segment. Each new segment starts at
  ``max(cursor, wall_clock_time)`` so that overlap is impossible even if
  timestamps are slightly off.
"""

import asyncio
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger


class AudioRecorder:
    """Records user input and TTS output into a dual-channel stereo audio file.

    Channel 1 (Left): User microphone input.
    Channel 2 (Right): TTS output (AI voice).

    A shared cursor guarantees no overlap between segments on either channel,
    while wall-clock timestamps preserve natural silence gaps between turns.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        """Initialize the audio recorder.

        Args:
            sample_rate: Sample rate for the audio recording in Hz (default: 16000).
        """
        self.sample_rate = sample_rate
        self._user_audio_segments: list[tuple[float, np.ndarray]] = []
        self._tts_audio_segments: list[tuple[float, np.ndarray]] = []
        self._cursor: float = 0.0  # End of last placed segment (seconds)
        self._start_time: float | None = None  # Wall-clock reference
        self._lock = asyncio.Lock()

        logger.debug(f"🎙️ AudioRecorder initialized with sample rate: {sample_rate} Hz")

    async def start(self) -> None:
        """Start the recording session.

        Sets the wall-clock reference and resets the cursor.
        """
        async with self._lock:
            if self._start_time is None:
                self._start_time = time.time()
                self._cursor = 0.0
                logger.info(f"🎙️ AudioRecorder started at {self._start_time}")

    def _relative_now(self) -> float:
        """Return seconds elapsed since recording started.

        Returns:
            Elapsed time in seconds, or 0.0 if not started.
        """
        if self._start_time is None:
            return 0.0
        return max(0.0, time.time() - self._start_time)

    async def add_user_audio(
        self, audio_data: np.ndarray, backdate_seconds: float = 0.0
    ) -> None:
        """Add user microphone audio to channel 1 (left).

        The segment is placed at ``max(cursor, wall_clock - backdate - duration)``
        so that natural silence gaps are preserved but overlap is impossible.

        Args:
            audio_data: Audio data as numpy array (float32).
            backdate_seconds: How many seconds to backdate the start of this
                segment (useful when the buffer was accumulated over time).
        """
        async with self._lock:
            if len(audio_data) == 0:
                return

            if self._start_time is None:
                self._start_time = time.time()

            duration = len(audio_data) / self.sample_rate
            # Where the audio *should* start based on wall clock
            wall_start = self._relative_now() - duration - backdate_seconds
            wall_start = max(0.0, wall_start)

            # Never overlap with previous segments
            timestamp = max(self._cursor, wall_start)

            self._user_audio_segments.append((timestamp, audio_data.astype(np.float32)))
            self._cursor = timestamp + duration

            logger.debug(
                f"📝 Added {len(audio_data)} user samples at {timestamp:.2f}s "
                f"(duration: {duration:.2f}s, cursor now: {self._cursor:.2f}s)"
            )

    async def add_tts_audio(
        self, audio_file_path: str, backdate_seconds: float = 0.0
    ) -> None:
        """Add TTS output audio to channel 2 (right).

        The segment is placed at ``max(cursor, wall_clock_now - backdate)`` so that
        silence gaps are preserved but overlap is impossible.

        Args:
            audio_file_path: Path to the TTS audio file.
            backdate_seconds: How many seconds to backdate the start of this
                segment (useful to account for TTS generation time).
        """
        async with self._lock:
            try:
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

                wall_now = self._relative_now()
                # Backdate the timestamp to account for TTS generation time
                timestamp = max(self._cursor, wall_now - backdate_seconds)
                duration = len(audio_data) / self.sample_rate

                self._tts_audio_segments.append(
                    (timestamp, audio_data.astype(np.float32))
                )
                self._cursor = timestamp + duration

                logger.debug(
                    f"📝 Added TTS audio from {audio_file_path} at {timestamp:.2f}s "
                    f"(duration: {duration:.2f}s, backdate: {backdate_seconds:.2f}s, "
                    f"cursor now: {self._cursor:.2f}s)"
                )

            except Exception as e:
                logger.error(f"❌ Error loading TTS audio file {audio_file_path}: {e}")

    async def handle_interruption(self, timestamp: float) -> None:
        """Handle interruption by truncating TTS audio beyond the interrupt point.

        Args:
            timestamp: The unix timestamp (time.time()) when interruption occurred.
        """
        async with self._lock:
            if self._start_time is None:
                return

            interrupt_rel = max(0.0, timestamp - self._start_time)

            logger.info(f"🛑 Handling interruption at {interrupt_rel:.2f}s")

            new_segments: list[tuple[float, np.ndarray]] = []
            for seg_start, audio in self._tts_audio_segments:
                seg_end = seg_start + (len(audio) / self.sample_rate)

                if seg_start >= interrupt_rel:
                    # Entirely in the future → remove
                    continue
                elif seg_end > interrupt_rel:
                    # Overlaps → truncate
                    keep_samples = int((interrupt_rel - seg_start) * self.sample_rate)
                    if keep_samples > 0:
                        new_segments.append((seg_start, audio[:keep_samples]))
                else:
                    # Entirely in the past → keep
                    new_segments.append((seg_start, audio))

            self._tts_audio_segments = new_segments

            # Rewind cursor to the actual end of remaining content
            self._cursor = self._compute_max_end()

    def _compute_max_end(self) -> float:
        """Compute the end time of the last segment across both channels.

        Returns:
            End time in seconds.
        """
        max_end = 0.0
        for seg_start, audio in self._user_audio_segments:
            max_end = max(max_end, seg_start + len(audio) / self.sample_rate)
        for seg_start, audio in self._tts_audio_segments:
            max_end = max(max_end, seg_start + len(audio) / self.sample_rate)
        return max_end

    def _resample(
        self, audio_data: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """Resample audio data to target sample rate using linear interpolation.

        Args:
            audio_data: Input audio data.
            orig_sr: Original sample rate.
            target_sr: Target sample rate.

        Returns:
            Resampled audio data.
        """
        if orig_sr == target_sr:
            return audio_data

        duration = len(audio_data) / orig_sr
        new_length = int(duration * target_sr)

        old_indices = np.linspace(0, len(audio_data) - 1, len(audio_data))
        new_indices = np.linspace(0, len(audio_data) - 1, new_length)

        resampled = np.interp(new_indices, old_indices, audio_data)

        logger.debug(
            f"🔄 Resampled audio from {orig_sr} Hz to {target_sr} Hz "
            f"({len(audio_data)} → {len(resampled)} samples)"
        )

        return resampled.astype(np.float32)

    def _build_timeline(
        self, segments: list[tuple[float, np.ndarray]], total_duration: float
    ) -> np.ndarray:
        """Build a complete audio timeline with proper timing from segments.

        Args:
            segments: List of (timestamp, audio_data) tuples.
            total_duration: Total duration of the recording in seconds.

        Returns:
            Complete audio array with silence padding.
        """
        total_samples = int(total_duration * self.sample_rate)
        timeline = np.zeros(total_samples, dtype=np.float32)

        for timestamp, audio_data in segments:
            start_sample = int(timestamp * self.sample_rate)
            end_sample = start_sample + len(audio_data)

            if end_sample > total_samples:
                end_sample = total_samples
                audio_data = audio_data[: end_sample - start_sample]

            if start_sample < total_samples:
                timeline[start_sample:end_sample] = audio_data

        return timeline

    async def save_recording(self, output_path: str) -> bool:
        """Save the dual-channel recording to a WAV file.

        User audio on channel 1 (left), TTS audio on channel 2 (right).

        Args:
            output_path: Path where the WAV file will be saved.

        Returns:
            True if save was successful, False otherwise.
        """
        async with self._lock:
            try:
                if not self._user_audio_segments and not self._tts_audio_segments:
                    logger.warning("⚠️ No audio data to save")
                    return False

                total_duration = self._compute_max_end()

                user_channel = self._build_timeline(
                    self._user_audio_segments, total_duration
                )
                tts_channel = self._build_timeline(
                    self._tts_audio_segments, total_duration
                )

                stereo_audio = np.stack([user_channel, tts_channel], axis=1)

                output_dir = Path(output_path).parent
                output_dir.mkdir(parents=True, exist_ok=True)

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

    async def save_recording_from_timestamp(
        self,
        output_path: str,
        start_timestamp: float,
        end_timestamp: float | None = None,
    ) -> bool:
        """Save a trimmed recording starting from a specific wall-clock timestamp.

        This is useful when video recording starts after audio recording has begun.
        The audio will be trimmed to align with the video start time and optionally
        padded to match a specific duration.

        Args:
            output_path: Path where the WAV file will be saved.
            start_timestamp: Unix timestamp (time.time()) when to start the recording.
            end_timestamp: Optional Unix timestamp when to end the recording.
                If provided, audio will be padded with silence to match this duration.

        Returns:
            True if save was successful, False otherwise.
        """
        async with self._lock:
            try:
                if self._start_time is None:
                    logger.warning("⚠️ Recording hasn't started yet")
                    return False

                # If we have no audio but have an end_timestamp, we can still create
                # a silent audio file to match the video duration
                has_audio = bool(self._user_audio_segments) or bool(
                    self._tts_audio_segments
                )
                if not has_audio and end_timestamp is None:
                    logger.warning("⚠️ No audio data to save and no duration specified")
                    return False

                # Calculate the offset from audio recording start
                trim_offset = max(0.0, start_timestamp - self._start_time)

                logger.info(
                    f"✂️ Trimming audio - audio_start: {self._start_time:.2f}, "
                    f"video_start: {start_timestamp:.2f}, "
                    f"video_end: {f'{end_timestamp:.2f}' if end_timestamp is not None else 'None'}, "
                    f"trim_offset: {trim_offset:.2f}s"
                )

                logger.debug(
                    f"📊 Before trimming - user segments: {len(self._user_audio_segments)}, "
                    f"tts segments: {len(self._tts_audio_segments)}"
                )
                if self._user_audio_segments:
                    for i, (ts, _) in enumerate(self._user_audio_segments[:3]):
                        logger.debug(f"  User segment {i}: timestamp={ts:.2f}s")
                if self._tts_audio_segments:
                    for i, (ts, _) in enumerate(self._tts_audio_segments[:3]):
                        logger.debug(f"  TTS segment {i}: timestamp={ts:.2f}s")

                # Filter and adjust segments to start from trim_offset
                def trim_segments(
                    segments: list[tuple[float, np.ndarray]],
                ) -> list[tuple[float, np.ndarray]]:
                    trimmed = []
                    for seg_start, audio_data in segments:
                        seg_end = seg_start + len(audio_data) / self.sample_rate

                        if seg_end <= trim_offset:
                            # Segment ends before trim point → skip
                            continue
                        elif seg_start >= trim_offset:
                            # Segment starts after trim point → shift timestamp
                            trimmed.append((seg_start - trim_offset, audio_data))
                        else:
                            # Segment overlaps trim point → split and shift
                            samples_to_skip = int(
                                (trim_offset - seg_start) * self.sample_rate
                            )
                            trimmed_audio = audio_data[samples_to_skip:]
                            trimmed.append((0.0, trimmed_audio))

                    return trimmed

                trimmed_user = trim_segments(self._user_audio_segments)
                trimmed_tts = trim_segments(self._tts_audio_segments)

                logger.debug(
                    f"📊 After trimming - user segments: {len(trimmed_user)}, "
                    f"tts segments: {len(trimmed_tts)}"
                )
                if trimmed_user:
                    for i, (ts, audio) in enumerate(trimmed_user[:3]):
                        duration = len(audio) / self.sample_rate
                        logger.debug(
                            f"  Trimmed user segment {i}: timestamp={ts:.2f}s, duration={duration:.2f}s"
                        )
                if trimmed_tts:
                    for i, (ts, audio) in enumerate(trimmed_tts[:3]):
                        duration = len(audio) / self.sample_rate
                        logger.debug(
                            f"  Trimmed TTS segment {i}: timestamp={ts:.2f}s, duration={duration:.2f}s"
                        )

                # Compute duration from audio segments
                max_end = 0.0
                for seg_start, audio in trimmed_user:
                    max_end = max(max_end, seg_start + len(audio) / self.sample_rate)
                for seg_start, audio in trimmed_tts:
                    max_end = max(max_end, seg_start + len(audio) / self.sample_rate)

                # If end_timestamp is provided, use it to determine the target duration
                # This ensures silence periods are preserved even when user hasn't spoken
                if end_timestamp is not None:
                    target_duration = max(0.0, end_timestamp - start_timestamp)
                    # Use the maximum of computed and target duration to preserve all audio
                    total_duration = max(max_end, target_duration)
                    logger.info(
                        f"📏 Target duration from timestamps: {target_duration:.2f}s, "
                        f"audio segments end at: {max_end:.2f}s, "
                        f"using: {total_duration:.2f}s"
                    )
                else:
                    total_duration = max_end
                    if total_duration == 0.0:
                        logger.warning(
                            f"⚠️ No audio data after trimming at offset {trim_offset:.2f}s"
                        )
                        return False

                # Build timelines with silence padding
                user_channel = self._build_timeline(trimmed_user, total_duration)
                tts_channel = self._build_timeline(trimmed_tts, total_duration)

                stereo_audio = np.stack([user_channel, tts_channel], axis=1)

                output_dir = Path(output_path).parent
                output_dir.mkdir(parents=True, exist_ok=True)

                sf.write(output_path, stereo_audio, self.sample_rate, subtype="PCM_16")

                logger.info(
                    f"💾 Trimmed recording saved to {output_path} "
                    f"(duration: {total_duration:.2f}s, offset: {trim_offset:.2f}s, "
                    f"user segments: {len(trimmed_user)}, "
                    f"tts segments: {len(trimmed_tts)})"
                )

                return True

            except Exception as e:
                logger.error(f"❌ Error saving trimmed recording to {output_path}: {e}")
                return False

    async def clear(self) -> None:
        """Clear all audio segments and reset the recorder."""
        async with self._lock:
            user_segments = len(self._user_audio_segments)
            tts_segments = len(self._tts_audio_segments)

            self._user_audio_segments = []
            self._tts_audio_segments = []
            self._cursor = 0.0
            self._start_time = None

            logger.debug(
                f"🧹 Cleared audio segments "
                f"(user: {user_segments}, tts: {tts_segments})"
            )

    def has_audio(self) -> bool:
        """Check if the recorder has any audio data.

        Returns:
            True if there is audio data in either channel, False otherwise.
        """
        return bool(self._user_audio_segments) or bool(self._tts_audio_segments)

    def get_duration(self) -> float:
        """Get the current duration of the recording in seconds.

        Returns:
            Duration in seconds based on the cursor position.
        """
        return self._cursor

    @staticmethod
    def generate_filename(
        conf_uid: str,
        history_uid: str | None = None,
        output_dir: str = "recordings",
    ) -> str:
        """Generate a filename for the recording.

        Args:
            conf_uid: Configuration UID.
            history_uid: Chat history UID (optional).
            output_dir: Output directory for recordings.

        Returns:
            Full path to the recording file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if history_uid:
            filename = f"recording_{conf_uid}_{history_uid}_{timestamp}.wav"
        else:
            filename = f"recording_{conf_uid}_{timestamp}.wav"

        return os.path.join(output_dir, filename)
