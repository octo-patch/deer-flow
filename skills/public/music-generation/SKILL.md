---
name: music-generation
description: Use this skill when the user requests to generate, create, compose, or produce music or songs — background music, theme songs, jingles, or instrumental tracks. Generates a song from a style/mood prompt and optional lyrics via the MiniMax music API.
---

# Music Generation Skill

## Overview

This skill generates songs (vocal or instrumental) from a structured JSON spec using the
MiniMax music generation API (`/v1/music_generation`). You describe the style/mood/scene in
`prompt`, optionally provide `lyrics`, and the script returns an audio file.

## Workflow

### Step 1: Understand Requirements

Identify the desired style, mood, scene, language, and whether the user wants vocals or a
pure instrumental track. Decide whether to supply lyrics or let the model write them.

### Step 2: Create the Spec JSON

Write a JSON file in `/mnt/user-data/workspace/` named `{descriptive-name}.json`:

```json
{
  "title": "Rainy Night Cafe",
  "prompt": "indie folk, melancholic, introspective, walking alone, cafe",
  "lyrics": "[verse]\nStreetlights glow the night wind sighs\n[chorus]\nPush the wooden door warm air inside"
}
```

Fields:
- `title` (optional): a human-readable name.
- `model` (optional): defaults to `music-3.0`.
- `prompt` (optional): style, mood, and scene. Drives the musical character.
- `lyrics` (optional): song lyrics. Use `\n` between lines and structure tags such as
  `[Intro]`, `[Verse]`, `[Pre Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]`.
- `is_instrumental` (optional, bool): set `true` for a pure instrumental track (no lyrics needed).
- `stream` (optional, bool): stream hex-encoded audio chunks. Defaults to `false`.
- `output_format` (optional): `hex` (default) or `url`. Streaming requires `hex`.
- `audio_setting` (optional): override `sample_rate`, `bitrate`, or `format`; supported formats
  are `mp3`, `wav`, and `pcm`.
- `aigc_watermark` (optional, bool): request a watermark when using the China endpoint.
- `audio_url` or `audio_base64` (optional): source audio for an audio-guided request.
- `cover_feature_id` (optional): a previously created cover feature to apply.

Behavior:
- `lyrics` provided → those lyrics are sung.
- `is_instrumental: true` → instrumental, no vocals.
- neither → the model auto-writes lyrics from `prompt` (`lyrics_optimizer`).

### Step 3: Execute Generation

```bash
python /mnt/skills/public/music-generation/scripts/generate.py \
  --prompt-file /mnt/user-data/workspace/rainy-night-cafe.json \
  --output-file /mnt/user-data/outputs/rainy-night-cafe.mp3
```

Parameters:
- `--prompt-file`: Absolute path to the JSON spec (required).
- `--output-file`: Absolute path for the generated audio (required).

[!NOTE]
Do NOT read the python file, just call it with the parameters.

## Environment

- `MINIMAX_API_KEY` (required): your MiniMax interface key.
- `MINIMAX_API_HOST` (optional): defaults to the China endpoint at `https://api.minimaxi.com`;
  use `https://api.minimax.io` for the global endpoint.
- `MINIMAX_MUSIC_MODEL` (optional): defaults to `music-3.0`; API-key free-tier users can
  select `music-3.0-free`. The current generation models are `music-3.0`, `music-2.6`,
  `music-3.0-free`, and `music-2.6-free`.

## Output Handling

- Music is saved in the requested MP3, WAV, or PCM format (typically in
  `/mnt/user-data/outputs/`). URL responses are downloaded before the command returns.
- Share the generated file with the user using the present_files tool.
- Offer to iterate on style or lyrics if adjustments are needed.

## Notes

- Keep `prompt` focused on style/mood/scene; put the actual sung words in `lyrics`.
- For non-English songs, write `lyrics` in the target language.
