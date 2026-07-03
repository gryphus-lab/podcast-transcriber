"""Main entry point for the podcast transcriber CLI."""

import argparse
import sys
from pathlib import Path

from rich.console import Console

from .config import HF_TOKEN, LANGUAGE, OUTPUT_DIR, SUPPORTED_FORMATS, WHISPER_MODEL
from .utils.formatter import format_transcript, write_transcript
from .utils.transcriber import transcribe_audio

console = Console()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe podcast audio with speaker diarization.",
    )
    parser.add_argument(
        "audio_file",
        type=Path,
        help="Path to the audio file (.m4a, .mp3, .wav, etc.)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=WHISPER_MODEL,
        help=f"Whisper model size (default: {WHISPER_MODEL})",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=LANGUAGE,
        help=f"Language code (default: {LANGUAGE})",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        default=True,
        help="Enable speaker diarization (default: enabled)",
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Disable speaker diarization (faster, no HF token needed)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=HF_TOKEN,
        help="HuggingFace token for diarization models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["txt", "srt", "json"],
        default="txt",
        help="Output format (default: txt)",
    )
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> bool:
    """Validate CLI inputs before processing."""
    if not args.audio_file.exists():
        console.print(f"[red]Error:[/red] File not found: {args.audio_file}")
        return False

    suffix = args.audio_file.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        console.print(
            f"[red]Error:[/red] Unsupported format '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
        return False

    if args.diarize and not args.no_diarize and not args.hf_token:
        console.print(
            "[yellow]Warning:[/yellow] No HF_TOKEN set. "
            "Speaker diarization requires a HuggingFace token.\n"
            "  Set HF_TOKEN in .env or pass --hf-token, "
            "or use --no-diarize to skip."
        )
        return False

    return True


def main() -> int:
    """CLI entry point."""
    args = parse_args()

    if not validate_inputs(args):
        return 1

    diarize = args.diarize and not args.no_diarize

    console.print("\n[bold]Podcast Transcriber[/bold]")
    console.print(f"  Audio:    {args.audio_file}")
    console.print(f"  Model:    {args.model}")
    console.print(f"  Language: {args.language}")
    console.print(f"  Diarize:  {'yes' if diarize else 'no'}")
    console.print(f"  Format:   {args.format}")
    console.print()

    try:
        # Transcribe
        with console.status("[bold green]Transcribing audio..."):
            result = transcribe_audio(
                audio_path=args.audio_file,
                model_name=args.model,
                language=args.language,
                diarize=diarize,
                hf_token=args.hf_token,
            )

        # Format and write output
        transcript = format_transcript(result, output_format=args.format)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = write_transcript(
            transcript,
            audio_path=args.audio_file,
            output_dir=args.output_dir,
            output_format=args.format,
        )

        console.print(f"\n[green]Done![/green] Transcript saved to: {output_path}")
        return 0

    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
