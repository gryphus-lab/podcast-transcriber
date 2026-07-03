"""Tests for transcript formatter."""

import json
from pathlib import Path

from podcast_transcriber.utils.formatter import (
    _seconds_to_srt_time,
    format_transcript,
    write_transcript,
)


class TestFormatTxt:
    """Test plain text formatting."""

    def test_basic_segments(self):
        result = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Hello there.", "speaker": "SPEAKER_00"},
                {"start": 2.5, "end": 5.0, "text": "How are you?", "speaker": "SPEAKER_00"},
                {"start": 5.0, "end": 8.0, "text": "I'm good, thanks.", "speaker": "SPEAKER_01"},
            ]
        }
        output = format_transcript(result, "txt")
        assert "[SPEAKER_00]" in output
        assert "[SPEAKER_01]" in output
        assert "Hello there." in output
        assert "I'm good, thanks." in output

    def test_empty_segments(self):
        result = {"segments": []}
        output = format_transcript(result, "txt")
        assert output.strip() == ""

    def test_no_speaker(self):
        result = {
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "No speaker here."},
            ]
        }
        output = format_transcript(result, "txt")
        assert "[Unknown]" in output


class TestFormatSrt:
    """Test SRT formatting."""

    def test_srt_structure(self):
        result = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "First line.", "speaker": "Host"},
                {"start": 3.0, "end": 5.5, "text": "Second line.", "speaker": "Guest"},
            ]
        }
        output = format_transcript(result, "srt")
        assert "1\n" in output
        assert "00:00:00,000 --> 00:00:02,500" in output
        assert "[Host] First line." in output
        assert "2\n" in output


class TestFormatJson:
    """Test JSON formatting."""

    def test_json_valid(self):
        result = {
            "segments": [
                {"start": 1.23, "end": 4.56, "text": "Test.", "speaker": "A"},
            ]
        }
        output = format_transcript(result, "json")
        parsed = json.loads(output)
        assert len(parsed) == 1
        assert parsed[0]["start"] == 1.23
        assert parsed[0]["speaker"] == "A"


class TestSecondsToSrtTime:
    """Test timestamp conversion."""

    def test_zero(self):
        assert _seconds_to_srt_time(0) == "00:00:00,000"

    def test_minutes(self):
        assert _seconds_to_srt_time(65.5) == "00:01:05,500"

    def test_hours(self):
        assert _seconds_to_srt_time(3661.123) == "01:01:01,123"


class TestWriteTranscript:
    """Test file writing."""

    def test_writes_file(self, tmp_path):
        transcript = "Hello world\n"
        audio_path = Path("my_podcast.m4a")
        output_path = write_transcript(
            transcript, audio_path, tmp_path, "txt"
        )
        assert output_path.exists()
        assert output_path.name == "my_podcast.txt"
        assert output_path.read_text() == "Hello world\n"
