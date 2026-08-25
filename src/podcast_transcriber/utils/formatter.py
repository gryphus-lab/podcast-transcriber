"""Transcript formatting and output writing."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

Segment = Mapping[str, Any]


def format_transcript(result: Mapping[str, Any], output_format: str = "txt") -> str:
    """Format transcription result into the desired output format.

    Args:
        result: WhisperX result dict with 'segments' list.
        output_format: One of 'txt', 'srt', 'json'.

    Returns:
        Formatted transcript string.
    """
    segments = result.get("segments", [])

    if output_format == "json":
        return _format_json(segments)
    if output_format == "srt":
        return _format_srt(segments)
    return _format_txt(segments)


def write_transcript(
    transcript: str,
    audio_path: Path,
    output_dir: Path,
    output_format: str = "txt",
) -> Path:
    """Write formatted transcript to file.

    Args:
        transcript: Formatted transcript string.
        audio_path: Original audio file path (used for naming).
        output_dir: Directory to write the output file.
        output_format: File extension to use.

    Returns:
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{audio_path.stem}.{output_format}"
    output_path.write_text(transcript, encoding="utf-8")
    return output_path


def _format_txt(segments: Sequence[Segment]) -> str:
    """Format segments as plain text with speaker labels."""
    lines = []
    current_speaker = None

    for segment in segments:
        speaker = segment.get("speaker", "Unknown")
        text = segment.get("text", "").strip()

        if not text:
            continue

        if speaker != current_speaker:
            current_speaker = speaker
            lines.append(f"\n[{speaker}]")

        lines.append(f"  {text}")

    return "\n".join(lines).strip() + "\n"


def _format_srt(segments: Sequence[Segment]) -> str:
    """Format segments as SRT subtitles."""
    lines = []

    for i, segment in enumerate(segments, start=1):
        start = _seconds_to_srt_time(segment.get("start", 0))
        end = _seconds_to_srt_time(segment.get("end", 0))
        speaker = segment.get("speaker", "")
        text = segment.get("text", "").strip()

        if not text:
            continue

        speaker_prefix = f"[{speaker}] " if speaker else ""
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(f"{speaker_prefix}{text}")
        lines.append("")

    return "\n".join(lines)


def _format_json(segments: Sequence[Segment]) -> str:
    """Format segments as JSON."""
    cleaned = []
    for segment in segments:
        entry = {
            "start": round(segment.get("start", 0), 2),
            "end": round(segment.get("end", 0), 2),
            "text": segment.get("text", "").strip(),
        }
        if "speaker" in segment:
            entry["speaker"] = segment["speaker"]
        if entry["text"]:
            cleaned.append(entry)

    return json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n"


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    total_millis = round(seconds * 1000)
    hours, remainder = divmod(total_millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
