from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class LatencyTracker:
    """Collects latency marks for a single conversation turn.

    Args:
        conversation_id: Unique identifier for the conversation.
        client_uid: Client unique identifier.
    """

    conversation_id: str
    client_uid: str
    start_time: float = field(default_factory=time.perf_counter)
    marks: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize the tracker with a conversation start mark."""
        self.marks["conversation_start"] = self.start_time

    def mark(self, name: str) -> None:
        """Record a timestamp for a named mark.

        Args:
            name: Name of the mark to record.
        """
        self.marks[name] = time.perf_counter()

    def mark_once(self, name: str) -> None:
        """Record a timestamp only if the mark is not set yet.

        Args:
            name: Name of the mark to record.
        """
        if name not in self.marks:
            self.marks[name] = time.perf_counter()

    def duration_ms(self, start: str, end: str) -> float | None:
        """Return elapsed milliseconds between two marks.

        Args:
            start: Start mark name.
            end: End mark name.

        Returns:
            Elapsed time in milliseconds, or None if marks are missing.
        """
        if start not in self.marks or end not in self.marks:
            return None
        return (self.marks[end] - self.marks[start]) * 1000.0

    def marks_ms(self) -> Dict[str, float]:
        """Return marks as milliseconds relative to conversation start.

        Returns:
            Mapping of mark name to milliseconds since conversation start.
        """
        return {
            name: (timestamp - self.start_time) * 1000.0
            for name, timestamp in self.marks.items()
        }

    def report(self) -> Dict[str, Any]:
        """Build a latency report dictionary.

        Returns:
            Dict with marks and derived durations.
        """
        durations = {
            "asr_ms": self.duration_ms("asr_start", "asr_end"),
            "llm_ttft_ms": self.duration_ms("llm_start", "llm_ttft"),
            "llm_total_ms": self.duration_ms("llm_start", "llm_end"),
            "tts_first_audio_ms": self.duration_ms(
                "tts_start", "tts_first_audio_ready"
            ),
            "tts_total_ms": self.duration_ms("tts_start", "tts_all_complete"),
            "first_payload_ms": self.duration_ms(
                "conversation_start", "tts_first_payload_sent"
            ),
            "backend_total_ms": self.duration_ms(
                "conversation_start", "backend_synth_complete"
            ),
            "playback_total_ms": self.duration_ms(
                "conversation_start", "playback_complete"
            ),
        }
        return {
            "conversation_id": self.conversation_id,
            "client_uid": self.client_uid,
            "marks_ms": self.marks_ms(),
            "durations_ms": durations,
        }

    def format_report(self) -> str:
        """Format the latency report for logging.

        Returns:
            Human-readable report string.
        """
        report = self.report()
        durations = report["durations_ms"]
        parts = [
            f"id={report['conversation_id']}",
            f"client={report['client_uid']}",
        ]
        for key, value in durations.items():
            if value is None:
                parts.append(f"{key}=n/a")
            else:
                parts.append(f"{key}={value:.1f}ms")
        return "Latency " + " ".join(parts)
