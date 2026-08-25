import argparse
import base64
import json
import os
from typing import Any

import requests

MINIMAX_DEFAULT_HOST = "https://api.minimax.io"
MINIMAX_MUSIC_PATH = "/v1/music_generation"


def _check_base_resp(payload: dict) -> None:
    base = payload.get("base_resp") or {}
    if base.get("status_code", 0) != 0:
        raise Exception(f"MiniMax error {base.get('status_code')}: {base.get('status_msg')}")


def _decode_audio(value: Any, output_format: str) -> bytes:
    """Decode an audio value returned as hex or base64."""
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str) or not value:
        raise Exception("MiniMax returned no audio data")
    if output_format == "base64":
        return base64.b64decode(value)
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise Exception("MiniMax returned invalid hex audio data") from exc


def _audio_from_payload(payload: dict, output_format: str) -> bytes | str:
    data = payload.get("data") or {}
    status = data.get("status")
    if status not in (1, 2):
        if not data:
            raise Exception("MiniMax returned no audio data (missing generation status)")
        raise Exception(f"MiniMax returned invalid generation status: {status}")
    audio = data.get("audio")
    if output_format == "url" or (isinstance(audio, str) and audio.startswith(("http://", "https://"))):
        if not audio:
            raise Exception("MiniMax returned no audio URL")
        return audio
    if audio:
        return _decode_audio(audio, output_format)
    # Some responses expose the encoded result under audio_base64.
    if data.get("audio_base64"):
        return _decode_audio(data["audio_base64"], "base64")
    raise Exception("MiniMax returned no audio data")


def _write_audio(value: bytes | str, output_file: str) -> None:
    if isinstance(value, str):
        download = requests.get(value, timeout=300)
        download.raise_for_status()
        content = download.content
    else:
        content = value
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "wb") as f:
        f.write(content)


def generate_music(prompt_file: str, output_file: str) -> str:
    """Generate a song from a JSON spec via MiniMax /v1/music_generation.

    Spec JSON supports prompt, lyrics, stream, output_format, audio_setting,
    lyrics_optimizer, is_instrumental, cover_feature_id, and aigc_watermark.
    - lyrics given        -> use them (supports [Verse]/[Chorus] structure tags, \\n lines)
    - is_instrumental true -> pure music, no lyrics needed
    - otherwise           -> lyrics_optimizer auto-writes lyrics from prompt
    """
    with open(prompt_file, "r", encoding="utf-8") as f:
        spec = json.load(f)

    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        return "MINIMAX_API_KEY is not set"

    prompt = (spec.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("`prompt` is required in the music spec")
    lyrics = spec.get("lyrics") or None  # treat empty string the same as absent
    is_instrumental = bool(spec.get("is_instrumental", False))

    output_format = str(spec.get("output_format") or "hex").lower()
    if output_format not in {"url", "hex", "base64"}:
        raise ValueError("output_format must be one of: url, hex, base64")
    audio_setting = spec.get("audio_setting")
    if audio_setting is None:
        audio_setting = {"sample_rate": 44100, "bitrate": 256000, "format": "mp3"}
    elif not isinstance(audio_setting, dict):
        raise ValueError("audio_setting must be an object")
    else:
        audio_setting = dict(audio_setting)
    body = {
        "model": os.getenv("MINIMAX_MUSIC_MODEL", "music-3.0"),
        "prompt": prompt,
        "output_format": output_format,
        "audio_setting": audio_setting,
    }
    if "stream" in spec:
        body["stream"] = bool(spec["stream"])
    if lyrics:
        body["lyrics"] = lyrics
    elif is_instrumental:
        body["is_instrumental"] = True
    else:
        body["lyrics_optimizer"] = bool(spec.get("lyrics_optimizer", True))
    if "lyrics_optimizer" in spec and lyrics:
        body["lyrics_optimizer"] = bool(spec["lyrics_optimizer"])
    if "cover_feature_id" in spec and spec["cover_feature_id"]:
        body["cover_feature_id"] = spec["cover_feature_id"]
    if "aigc_watermark" in spec:
        body["aigc_watermark"] = bool(spec["aigc_watermark"])

    host = os.getenv("MINIMAX_API_HOST", MINIMAX_DEFAULT_HOST).rstrip("/")
    response = requests.post(
        f"{host}{MINIMAX_MUSIC_PATH}",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        stream=bool(body.get("stream", False)),
        timeout=300,
    )
    response.raise_for_status()
    if body.get("stream"):
        payload = None
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if isinstance(raw_line, bytes):
                raw_line = raw_line.decode("utf-8", errors="replace")
            line = raw_line.removeprefix("data:").strip()
            if line == "[DONE]":
                continue
            try:
                candidate = json.loads(line)
            except (TypeError, json.JSONDecodeError):
                continue
            _check_base_resp(candidate)
            payload = candidate
            if (candidate.get("data") or {}).get("status") == 2:
                break
        if payload is None:
            raise Exception("MiniMax returned no streamed audio data")
    else:
        payload = response.json()
        _check_base_resp(payload)
    _check_base_resp(payload)
    _write_audio(_audio_from_payload(payload, output_format), output_file)
    return f"Successfully generated music to {output_file}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate music using MiniMax API")
    parser.add_argument("--prompt-file", required=True,
                        help="Absolute path to JSON spec file {title, prompt, lyrics?, is_instrumental?}")
    parser.add_argument("--output-file", required=True, help="Output path for generated MP3")
    args = parser.parse_args()

    try:
        print(generate_music(args.prompt_file, args.output_file))
    except Exception as e:
        print(f"Error while generating music: {e}")
