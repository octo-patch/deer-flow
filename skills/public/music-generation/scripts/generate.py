import argparse
import json
import os

import requests

MINIMAX_DEFAULT_HOST = "https://api.minimaxi.com"
MINIMAX_DEFAULT_MODEL = "music-3.0"
DEFAULT_AUDIO_SETTING = {"sample_rate": 44100, "bitrate": 256000, "format": "mp3"}
SUPPORTED_AUDIO_FORMATS = {"mp3", "wav", "pcm"}
SUPPORTED_OUTPUT_FORMATS = {"hex", "url"}
IN_PROGRESS_STATUS = 1
COMPLETED_STATUS = 2


def _check_base_resp(payload: dict) -> None:
    base = payload.get("base_resp") or {}
    if base.get("status_code") != 0:
        raise Exception(
            f"MiniMax error {base.get('status_code')}: {base.get('status_msg')}"
        )


def _response_data(payload: dict, require_completed: bool) -> dict:
    _check_base_resp(payload)
    data = payload.get("data") or {}
    status = data.get("status")
    if status not in {IN_PROGRESS_STATUS, COMPLETED_STATUS}:
        raise Exception(f"MiniMax returned an invalid music status: {status}")
    if require_completed and status != COMPLETED_STATUS:
        raise Exception(f"MiniMax music generation did not complete: status {status}")
    return data


def _audio_bytes(
    response: requests.Response, output_format: str, stream: bool
) -> bytes:
    if stream:
        chunks = []
        final_status = None
        for raw_line in response.iter_lines():
            line = raw_line.decode("utf-8").strip()
            if not line or line in {"data: [DONE]", "data:[DONE]"}:
                continue
            if line.startswith("data:"):
                line = line[5:].lstrip()
            payload = json.loads(line)
            data = _response_data(payload, require_completed=False)
            final_status = data["status"]
            audio = data.get("audio")
            if audio:
                chunks.append(bytes.fromhex(audio))
        if final_status != COMPLETED_STATUS:
            raise Exception("MiniMax streaming music generation did not complete")
        if not chunks:
            raise Exception("MiniMax returned no audio data")
        return b"".join(chunks)

    payload = response.json()
    audio = _response_data(payload, require_completed=True).get("audio")
    if not audio:
        raise Exception("MiniMax returned no audio data")
    if output_format == "hex":
        return bytes.fromhex(audio)

    download = requests.get(audio, timeout=300)
    download.raise_for_status()
    return download.content


def generate_music(prompt_file: str, output_file: str) -> str:
    """Generate a song from a JSON spec via MiniMax /v1/music_generation.

    Spec JSON: {"title": str, "prompt": str, "lyrics"?: str, "is_instrumental"?: bool,
                "stream"?: bool, "output_format"?: "hex" | "url", "audio_setting"?: dict}
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
    lyrics = spec.get("lyrics") or None  # treat empty string the same as absent
    is_instrumental = bool(spec.get("is_instrumental", False))
    stream = bool(spec.get("stream", False))
    output_format = spec.get("output_format", "hex")
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError("`output_format` must be `hex` or `url`")
    if stream and output_format != "hex":
        raise ValueError(
            "streaming music generation requires `output_format` to be `hex`"
        )

    audio_overrides = spec.get("audio_setting") or {}
    if not isinstance(audio_overrides, dict):
        raise ValueError("`audio_setting` must be an object")
    audio_setting = DEFAULT_AUDIO_SETTING | audio_overrides
    if audio_setting["format"] not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError("`audio_setting.format` must be `mp3`, `wav`, or `pcm`")

    body = {
        "model": spec.get("model")
        or os.getenv("MINIMAX_MUSIC_MODEL", MINIMAX_DEFAULT_MODEL),
        "stream": stream,
        "output_format": output_format,
        "audio_setting": audio_setting,
    }
    if prompt:
        body["prompt"] = prompt
    for field in ("audio_url", "audio_base64", "cover_feature_id"):
        if spec.get(field):
            body[field] = spec[field]
    if "aigc_watermark" in spec:
        body["aigc_watermark"] = bool(spec["aigc_watermark"])
    if lyrics:
        body["lyrics"] = lyrics
    elif is_instrumental:
        body["is_instrumental"] = True
    else:
        body["lyrics_optimizer"] = True

    host = os.getenv("MINIMAX_API_HOST", MINIMAX_DEFAULT_HOST).rstrip("/")
    response = requests.post(
        f"{host}/v1/music_generation",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=300,
        stream=stream,
    )
    response.raise_for_status()
    audio = _audio_bytes(response, output_format, stream)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "wb") as f:
        f.write(audio)
    return f"Successfully generated music to {output_file}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate music using MiniMax API")
    parser.add_argument(
        "--prompt-file",
        required=True,
        help="Absolute path to JSON spec file {title, prompt, lyrics?, is_instrumental?}",
    )
    parser.add_argument(
        "--output-file", required=True, help="Output path for generated audio"
    )
    args = parser.parse_args()

    try:
        print(generate_music(args.prompt_file, args.output_file))
    except Exception as e:
        print(f"Error while generating music: {e}")
