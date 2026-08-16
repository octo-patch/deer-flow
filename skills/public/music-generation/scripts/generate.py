import argparse
import base64
import json
import os

import requests

# Music generation is served from two regional endpoints with the same API surface,
# apart from the region-only request fields listed in REGIONAL_FIELDS.
MINIMAX_HOSTS = {
    "global": "https://api.minimax.io",
    "cn": "https://api.minimaxi.com",
}
MINIMAX_REGION_ALIASES = {
    "global": "global",
    "global_en": "global",
    "en": "global",
    "cn": "cn",
    "cn_zh": "cn",
    "zh": "cn",
}
MINIMAX_DEFAULT_REGION = "cn"
MINIMAX_REGION_BY_HOST = {host: region for region, host in MINIMAX_HOSTS.items()}
MINIMAX_DEFAULT_MODEL = "music-3.0"
# Request fields accepted by one region only.
REGIONAL_FIELDS = {"global": frozenset(), "cn": frozenset({"aigc_watermark"})}

OUTPUT_FORMATS = ("url", "hex")
DEFAULT_OUTPUT_FORMAT = "hex"
# Streamed responses carry hex fragments; `url` output is rejected while streaming.
STREAM_OUTPUT_FORMATS = ("hex",)

AUDIO_FORMATS = ("mp3", "wav", "pcm")
SAMPLE_RATES = (16000, 24000, 32000, 44100)
BITRATES = (32000, 64000, 128000, 256000)
DEFAULT_AUDIO_SETTING = {"sample_rate": 44100, "bitrate": 256000, "format": "mp3"}

# data.status: the track is still being synthesized (1) or finished (2).
STATUS_IN_PROGRESS = 1
STATUS_COMPLETED = 2

# Reference audio inputs, accepted by the cover models only.
COVER_INPUT_FIELDS = ("audio_url", "audio_base64", "audio_file", "cover_feature_id")
COVER_INPUT_MIN_SECONDS = 6
COVER_INPUT_MAX_SECONDS = 360
COVER_INPUT_MAX_BYTES = 50 * 1024 * 1024


def _check_base_resp(payload: dict) -> None:
    base = payload.get("base_resp") or {}
    if base.get("status_code", 0) != 0:
        raise Exception(f"MiniMax error {base.get('status_code')}: {base.get('status_msg')}")


def _check_status(status) -> None:
    """Validate data.status; a track is only usable once it reports STATUS_COMPLETED."""
    if status is None or status == STATUS_COMPLETED:
        return
    if status == STATUS_IN_PROGRESS:
        raise Exception("MiniMax music generation is still in progress; retry the request")
    raise Exception(f"MiniMax returned an unexpected data.status: {status}")


def _resolve_endpoint() -> tuple[str, str]:
    """Resolve the (region, host) pair to call.

    MINIMAX_API_REGION picks a documented regional endpoint and MINIMAX_API_HOST overrides
    the host, e.g. for a private gateway. When only the host is set, the region is derived
    from it so that the region-only request fields stay correct.
    """
    raw_region = (os.getenv("MINIMAX_API_REGION") or "").strip().lower()
    host = (os.getenv("MINIMAX_API_HOST") or "").rstrip("/")
    if raw_region:
        region = MINIMAX_REGION_ALIASES.get(raw_region)
        if region is None:
            raise ValueError(
                f"Unknown MiniMax region {raw_region!r} "
                f"(use one of: {', '.join(sorted(MINIMAX_REGION_ALIASES))})"
            )
    else:
        region = MINIMAX_REGION_BY_HOST.get(host, MINIMAX_DEFAULT_REGION)
    return region, host or MINIMAX_HOSTS[region]


def _require_choice(field: str, value, allowed) -> None:
    if value not in allowed:
        raise ValueError(
            f"`{field}` must be one of {', '.join(str(item) for item in allowed)}, got {value!r}"
        )


def _is_cover_model(model: str) -> bool:
    """Cover models rework reference audio instead of composing from scratch, so they
    take the reference inputs and reject the lyrics_optimizer / is_instrumental flags.
    """
    return model.startswith("music-cover")


def _audio_setting(spec: dict) -> dict:
    """Merge the spec's audio_setting over the defaults and validate every value."""
    override = spec.get("audio_setting") or {}
    if not isinstance(override, dict):
        raise ValueError("`audio_setting` must be an object")
    setting = {**DEFAULT_AUDIO_SETTING, **{k: v for k, v in override.items() if v is not None}}
    unknown = sorted(set(setting) - set(DEFAULT_AUDIO_SETTING))
    if unknown:
        raise ValueError(f"Unknown `audio_setting` field(s): {', '.join(unknown)}")
    _require_choice("audio_setting.sample_rate", setting["sample_rate"], SAMPLE_RATES)
    _require_choice("audio_setting.bitrate", setting["bitrate"], BITRATES)
    _require_choice("audio_setting.format", setting["format"], AUDIO_FORMATS)
    return setting


def _encode_audio_file(path: str) -> str:
    """Base64-encode a local reference track for the audio_base64 input."""
    size = os.path.getsize(path)
    if size > COVER_INPUT_MAX_BYTES:
        raise ValueError(
            f"Reference audio is {size} bytes but the limit is {COVER_INPUT_MAX_BYTES} bytes "
            f"({COVER_INPUT_MIN_SECONDS}-{COVER_INPUT_MAX_SECONDS} seconds of audio)"
        )
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _cover_inputs(spec: dict, lyrics: str | None) -> dict:
    """Return the one reference-audio field a cover request carries.

    Exactly one of audio_url, audio_base64 or cover_feature_id is allowed; a local
    `audio_file` path is read here and sent as audio_base64.
    """
    provided = [name for name in COVER_INPUT_FIELDS if spec.get(name)]
    if len(provided) != 1:
        raise ValueError(
            "A cover model needs exactly one reference input: `audio_url`, `audio_base64` "
            f"(or a local `audio_file`) or `cover_feature_id`; got {len(provided)}"
        )
    name = provided[0]
    if name == "cover_feature_id" and not lyrics:
        raise ValueError("`lyrics` is required when a cover request uses `cover_feature_id`")
    if name == "audio_file":
        return {"audio_base64": _encode_audio_file(spec["audio_file"])}
    return {name: spec[name]}


def _build_body(spec: dict, region: str) -> dict:
    """Build the /v1/music_generation request body from the spec."""
    model = (spec.get("model") or os.getenv("MINIMAX_MUSIC_MODEL") or MINIMAX_DEFAULT_MODEL).strip()
    prompt = (spec.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("`prompt` is required in the music spec")
    stream = bool(spec.get("stream", False))
    output_format = (spec.get("output_format") or DEFAULT_OUTPUT_FORMAT).strip().lower()
    _require_choice(
        "output_format", output_format, STREAM_OUTPUT_FORMATS if stream else OUTPUT_FORMATS
    )

    body = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "output_format": output_format,
        "audio_setting": _audio_setting(spec),
    }
    lyrics = spec.get("lyrics") or None  # treat empty string the same as absent
    if lyrics:
        body["lyrics"] = lyrics

    if _is_cover_model(model):
        body.update(_cover_inputs(spec, lyrics))
    else:
        for name in COVER_INPUT_FIELDS:
            if spec.get(name):
                raise ValueError(f"`{name}` requires a cover model, but `model` is {model!r}")
        is_instrumental = bool(spec.get("is_instrumental", False))
        if is_instrumental:
            body["is_instrumental"] = True
        optimizer = spec.get("lyrics_optimizer")
        if optimizer is None:
            # Nothing to sing and no instrumental request: let the model write the lyrics.
            optimizer = not lyrics and not is_instrumental
        if optimizer:
            body["lyrics_optimizer"] = True

    watermark = spec.get("aigc_watermark")
    if watermark is not None:
        if "aigc_watermark" not in REGIONAL_FIELDS[region]:
            raise ValueError(f"`aigc_watermark` is not available on the {region} endpoint")
        if stream:
            raise ValueError("`aigc_watermark` only applies to non-streaming requests")
        body["aigc_watermark"] = bool(watermark)
    return body


def _download(url: str) -> bytes:
    """Fetch a `url` result immediately: the link expires 24 hours after generation."""
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    return response.content


def _audio_from_payload(payload: dict, output_format: str) -> bytes:
    """Read data.audio from a non-streaming response as a URL or a hex string."""
    _check_base_resp(payload)
    data = payload.get("data") or {}
    _check_status(data.get("status"))
    audio = data.get("audio")
    if not audio:
        raise Exception("MiniMax returned no audio data")
    if output_format == "url":
        return _download(audio)
    return bytes.fromhex(audio)


def _audio_from_stream(response) -> bytes:
    """Concatenate the hex fragments streamed as `data:`-prefixed events."""
    fragments: list[str] = []
    last_status = None
    for raw_line in response.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if line.startswith("data:"):
            line = line[len("data:"):].strip()
        if not line.startswith("{"):  # keep-alive comments and end-of-stream sentinels
            continue
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        _check_base_resp(payload)
        data = payload.get("data") or {}
        if data.get("status") is not None:
            last_status = data["status"]
        if data.get("audio"):
            fragments.append(data["audio"])
    if not fragments:
        raise Exception("MiniMax returned no audio data")
    if last_status == STATUS_IN_PROGRESS:
        raise Exception("The MiniMax music stream ended while the track was still in progress")
    return bytes.fromhex("".join(fragments))


def generate_music(prompt_file: str, output_file: str) -> str:
    """Generate a song from a JSON spec via MiniMax /v1/music_generation.

    Spec JSON: {"title"?, "model"?, "prompt", "lyrics"?, "stream"?, "output_format"?,
                "audio_setting"?, "lyrics_optimizer"?, "is_instrumental"?,
                "aigc_watermark"?, "audio_url"? | "audio_base64"? | "audio_file"?
                | "cover_feature_id"?}
    - lyrics given         -> use them (supports [Verse]/[Chorus] structure tags, \\n lines)
    - is_instrumental true -> pure music, no lyrics needed
    - otherwise            -> lyrics_optimizer auto-writes lyrics from prompt
    - cover model          -> rework one reference track instead of composing
    """
    with open(prompt_file, "r", encoding="utf-8") as f:
        spec = json.load(f)

    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        return "MINIMAX_API_KEY is not set"

    region, host = _resolve_endpoint()
    body = _build_body(spec, region)
    stream = body["stream"]
    response = requests.post(
        f"{host}/v1/music_generation",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=300,
        stream=stream,
    )
    response.raise_for_status()
    if stream:
        audio = _audio_from_stream(response)
    else:
        audio = _audio_from_payload(response.json(), body["output_format"])

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "wb") as f:
        f.write(audio)
    return f"Successfully generated music to {output_file}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate music using MiniMax API")
    parser.add_argument("--prompt-file", required=True,
                        help="Absolute path to the JSON music spec file")
    parser.add_argument("--output-file", required=True,
                        help="Output path; use the extension of audio_setting.format")
    args = parser.parse_args()

    try:
        print(generate_music(args.prompt_file, args.output_file))
    except Exception as e:
        print(f"Error while generating music: {e}")
