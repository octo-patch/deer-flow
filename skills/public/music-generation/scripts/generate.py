import argparse
import base64
import json
import os

import requests

MUSIC_ENDPOINT_PATH = "/v1/music_generation"

# The music endpoint is served from two regional hosts. Both expose the same
# contract; only the China host accepts the regional fields listed below.
MINIMAX_HOSTS = {
    "global": "https://api.minimax.io",
    "cn": "https://api.minimaxi.com",
}
DEFAULT_REGION = "cn"
MINIMAX_DEFAULT_HOST = MINIMAX_HOSTS[DEFAULT_REGION]
REGIONAL_ONLY_FIELDS = {"global": (), "cn": ("aigc_watermark",)}

DEFAULT_MUSIC_MODEL = "music-3.0"
GENERATION_MODELS = ("music-3.0", "music-2.6", "music-3.0-free", "music-2.6-free")
COVER_MODELS = ("music-cover", "music-cover-free")

# `output_format` defaults to hex; streaming responses only carry hex chunks.
OUTPUT_FORMATS = ("url", "hex")
DEFAULT_OUTPUT_FORMAT = "hex"
STREAM_OUTPUT_FORMATS = ("hex",)
# Returned url links expire after this many hours, so download immediately.
URL_TTL_HOURS = 24

AUDIO_FORMATS = ("mp3", "wav", "pcm")
DEFAULT_AUDIO_SETTING = {"sample_rate": 44100, "bitrate": 256000, "format": "mp3"}

# data.status reports synthesis progress, not the transport status.
STATUS_IN_PROGRESS = 1
STATUS_COMPLETED = 2

# Cover models take exactly one reference input: a URL, inline audio, or a
# feature id produced by the cover preprocess step.
COVER_INPUT_FIELDS = ("audio_url", "audio_base64", "cover_feature_id")
COVER_INPUT_MIN_SECONDS = 6
COVER_INPUT_MAX_SECONDS = 360
COVER_INPUT_MAX_MB = 50


def _check_base_resp(payload: dict) -> None:
    base = payload.get("base_resp") or {}
    if base.get("status_code", 0) != 0:
        raise Exception(f"MiniMax error {base.get('status_code')}: {base.get('status_msg')}")


def _resolve_region() -> str:
    region = (os.getenv("MINIMAX_MUSIC_REGION") or DEFAULT_REGION).strip().lower()
    if region not in MINIMAX_HOSTS:
        raise ValueError(
            f"Unknown MINIMAX_MUSIC_REGION {region!r} "
            f"(use one of {', '.join(sorted(MINIMAX_HOSTS))})"
        )
    return region


def _minimax_host(region: str) -> str:
    """Regional host, still overridable end to end with MINIMAX_API_HOST."""
    return os.getenv("MINIMAX_API_HOST", MINIMAX_HOSTS[region]).rstrip("/")


def _ensure_output_dir(output_file: str) -> None:
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def _resolve_audio_setting(spec: dict) -> dict:
    audio_setting = dict(DEFAULT_AUDIO_SETTING)
    override = spec.get("audio_setting") or {}
    if not isinstance(override, dict):
        raise ValueError("`audio_setting` must be an object in the music spec")
    audio_setting.update(override)
    audio_format = str(audio_setting.get("format") or "").strip().lower()
    if audio_format not in AUDIO_FORMATS:
        raise ValueError(
            f"`audio_setting.format` must be one of {', '.join(AUDIO_FORMATS)}"
        )
    audio_setting["format"] = audio_format
    return audio_setting


def _resolve_cover_inputs(spec: dict) -> dict:
    inputs = {field: spec[field] for field in COVER_INPUT_FIELDS if spec.get(field)}
    encoded = inputs.get("audio_base64")
    if encoded:
        try:
            size_mb = len(base64.b64decode(encoded)) / (1024 * 1024)
        except Exception as exc:
            raise ValueError(f"`audio_base64` is not valid base64 audio: {exc}") from exc
        if size_mb > COVER_INPUT_MAX_MB:
            raise ValueError(
                f"`audio_base64` holds {size_mb:.1f} MB; the reference audio limit "
                f"is {COVER_INPUT_MAX_MB} MB"
            )
    return inputs


def _build_body(spec: dict, region: str) -> dict:
    """Assemble the documented music generation request body."""
    model = (os.getenv("MINIMAX_MUSIC_MODEL") or DEFAULT_MUSIC_MODEL).strip()
    if model not in GENERATION_MODELS + COVER_MODELS:
        raise ValueError(
            f"Unknown music model {model!r} (use one of "
            f"{', '.join(GENERATION_MODELS + COVER_MODELS)})"
        )
    is_cover = model in COVER_MODELS

    prompt = (spec.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("`prompt` is required in the music spec")
    lyrics = (spec.get("lyrics") or "").strip() or None  # empty string == absent

    stream = bool(spec.get("stream", False))
    output_format = str(spec.get("output_format") or DEFAULT_OUTPUT_FORMAT).strip().lower()
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"`output_format` must be one of {', '.join(OUTPUT_FORMATS)}")
    if stream and output_format not in STREAM_OUTPUT_FORMATS:
        raise ValueError(
            "`output_format` must be "
            f"{' or '.join(STREAM_OUTPUT_FORMATS)} when `stream` is true"
        )

    body = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "output_format": output_format,
        "audio_setting": _resolve_audio_setting(spec),
    }

    cover_inputs = _resolve_cover_inputs(spec)
    if is_cover:
        if len(cover_inputs) != 1:
            raise ValueError(
                "cover models need exactly one of "
                f"{', '.join(COVER_INPUT_FIELDS)} in the music spec; reference "
                f"audio must run {COVER_INPUT_MIN_SECONDS}-{COVER_INPUT_MAX_SECONDS} "
                f"seconds and stay under {COVER_INPUT_MAX_MB} MB"
            )
        if "cover_feature_id" in cover_inputs and not lyrics:
            raise ValueError("`lyrics` is required when `cover_feature_id` is used")
        body.update(cover_inputs)
        if lyrics:
            body["lyrics"] = lyrics
    else:
        if cover_inputs:
            raise ValueError(
                f"{', '.join(COVER_INPUT_FIELDS)} are only accepted by the cover "
                f"models ({', '.join(COVER_MODELS)})"
            )
        if lyrics:
            body["lyrics"] = lyrics
        elif spec.get("is_instrumental", False):
            body["is_instrumental"] = True
        else:
            body["lyrics_optimizer"] = True

    watermark = spec.get("aigc_watermark")
    if watermark is not None:
        if "aigc_watermark" not in REGIONAL_ONLY_FIELDS[region]:
            raise ValueError(
                f"`aigc_watermark` is not accepted by the {region!r} music endpoint"
            )
        if stream:
            raise ValueError("`aigc_watermark` only applies when `stream` is false")
        body["aigc_watermark"] = bool(watermark)

    return body


def _audio_from_payload(payload: dict) -> str:
    """Read data.audio once data.status reports a completed synthesis."""
    data = payload.get("data") or {}
    status = data.get("status")
    if status is not None and status != STATUS_COMPLETED:
        if status == STATUS_IN_PROGRESS:
            raise Exception(
                f"MiniMax music synthesis is still in progress (data.status={status})"
            )
        raise Exception(f"MiniMax returned an unexpected data.status={status}")
    audio = data.get("audio")
    if not audio:
        raise Exception("MiniMax returned no audio data")
    return audio


def _audio_from_stream(response) -> str:
    """Join the hex audio chunks of a streaming music response.

    Streamed chunks carry the same documented `data.audio`, `data.status` and
    `base_resp` fields as a non-streaming response, one JSON object per event.
    """
    chunks = []
    completed = False
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if line.startswith("data:"):
            line = line[len("data:"):].strip()
        if not line or line == "[DONE]":
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip keep-alive and comment lines
        _check_base_resp(payload)
        data = payload.get("data") or {}
        if data.get("audio"):
            chunks.append(data["audio"])
        if data.get("status") == STATUS_COMPLETED:
            completed = True
    if not chunks:
        raise Exception("MiniMax returned no audio data")
    if not completed:
        raise Exception(
            "MiniMax music stream ended before data.status reported completion"
        )
    return "".join(chunks)


def _write_audio(audio: str, output_format: str, output_file: str) -> None:
    if output_format == "url":
        # The link is short-lived, so fetch it right away.
        response = requests.get(audio, timeout=300)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise Exception(
                "Could not download the generated audio; url links expire after "
                f"{URL_TTL_HOURS} hours: {exc}"
            ) from exc
        content = response.content
    else:
        content = bytes.fromhex(audio)
    _ensure_output_dir(output_file)
    with open(output_file, "wb") as f:
        f.write(content)


def generate_music(prompt_file: str, output_file: str) -> str:
    """Generate a song from a JSON spec via MiniMax /v1/music_generation.

    Spec JSON: {"title": str, "prompt": str, "lyrics"?: str,
    "is_instrumental"?: bool, "stream"?: bool, "output_format"?: "url"|"hex",
    "audio_setting"?: object, "aigc_watermark"?: bool} plus exactly one of
    audio_url / audio_base64 / cover_feature_id for the cover models.
    - lyrics given        -> use them (supports [Verse]/[Chorus] structure tags, \\n lines)
    - is_instrumental true -> pure music, no lyrics needed
    - otherwise           -> lyrics_optimizer auto-writes lyrics from prompt
    """
    with open(prompt_file, "r", encoding="utf-8") as f:
        spec = json.load(f)

    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        return "MINIMAX_API_KEY is not set"

    region = _resolve_region()
    body = _build_body(spec, region)
    stream = body["stream"]

    response = requests.post(
        f"{_minimax_host(region)}{MUSIC_ENDPOINT_PATH}",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=300,
        stream=stream,
    )
    response.raise_for_status()
    if stream:
        audio = _audio_from_stream(response)
    else:
        payload = response.json()
        _check_base_resp(payload)
        audio = _audio_from_payload(payload)

    _write_audio(audio, body["output_format"], output_file)
    return f"Successfully generated music to {output_file}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate music using MiniMax API")
    parser.add_argument("--prompt-file", required=True,
                        help="Absolute path to JSON spec file {title, prompt, lyrics?, ...}")
    parser.add_argument("--output-file", required=True,
                        help="Output path for the generated audio file")
    args = parser.parse_args()

    try:
        print(generate_music(args.prompt_file, args.output_file))
    except Exception as e:
        print(f"Error while generating music: {e}")
