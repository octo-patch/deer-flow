---
name: music-generation
description: Use this skill when the user requests to generate, create, compose, or produce music or songs — background music, theme songs, jingles, or instrumental tracks. Generates a song from a style/mood prompt and optional lyrics via the MiniMax music API.
---

# Music Generation Skill

## Overview

This skill generates songs (vocal, instrumental, or a cover of a reference track) from a
structured JSON spec using the MiniMax music generation API (`/v1/music_generation`). You
describe the style/mood/scene in `prompt`, optionally provide `lyrics`, and the script writes
the audio file.

## Workflow

### Step 1: Understand Requirements

Identify the desired style, mood, scene, language, and whether the user wants vocals, a pure
instrumental track, or a cover of an existing recording. Decide whether to supply lyrics or
let the model write them.

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
- `prompt` (required): style, mood, and scene. Drives the musical character.
- `lyrics` (optional): song lyrics. Use `\n` between lines and structure tags such as
  `[Intro]`, `[Verse]`, `[Pre Chorus]`, `[Chorus]`, `[Bridge]`, `[Outro]`.
- `is_instrumental` (optional, bool): set `true` for a pure instrumental track (no lyrics needed).
- `lyrics_optimizer` (optional, bool): let the model write or polish the lyrics. Enabled
  automatically when neither `lyrics` nor `is_instrumental` is given.
- `model` (optional): overrides `MINIMAX_MUSIC_MODEL` for this request.
- `stream` (optional, bool): stream the audio back in fragments. Only `hex` output is
  supported while streaming, and the fragments are reassembled into one file.
- `output_format` (optional): `hex` (default) returns the audio inline; `url` returns a link
  that the script downloads immediately, because the link expires after 24 hours.
- `audio_setting` (optional object):
  - `format`: `mp3` (default), `wav`, or `pcm`.
  - `sample_rate`: `16000`, `24000`, `32000`, or `44100` (default).
  - `bitrate`: `32000`, `64000`, `128000`, or `256000` (default).
- `aigc_watermark` (optional, bool): append a watermark to the audio. Available on the
  China endpoint only, and non-streaming requests only.

Cover requests (set `model` to a cover model) rework one reference track. Supply exactly one of:
- `audio_url`: link to the reference recording.
- `audio_base64`: the reference recording, already base64-encoded.
- `audio_file`: absolute path to a local reference recording; the script encodes it.
- `cover_feature_id`: a pre-processed reference id; `lyrics` is required with it.

Reference audio must be 6 seconds to 6 minutes long and at most 50 MB. Without `lyrics`, a
cover reuses the words detected in the reference recording.

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
- `--output-file`: Absolute path for the output audio file (required). Use the extension that
  matches `audio_setting.format`.

[!NOTE]
Do NOT read the python file, just call it with the parameters.

## Environment

- `MINIMAX_API_KEY` (required): your MiniMax interface key.
- `MINIMAX_API_REGION` (optional): `cn` (default, `https://api.minimaxi.com`) or `global`
  (`https://api.minimax.io`). Use the region that issued the key.
- `MINIMAX_API_HOST` (optional): overrides the regional host, e.g. for a private gateway.
- `MINIMAX_MUSIC_MODEL` (optional): default `music-3.0`. Generation models are `music-3.0`,
  `music-2.6`, `music-3.0-free` and `music-2.6-free`; cover models are `music-cover` and
  `music-cover-free`. The `-free` models work for all API-key users; the others need a
  paid/Token plan and allow a higher request rate.

## Output Handling

- Music is saved in the requested format (typically in `/mnt/user-data/outputs/`).
- Share the generated file with the user using the present_files tool.
- Offer to iterate on style or lyrics if adjustments are needed.

## Notes

- Keep `prompt` focused on style/mood/scene; put the actual sung words in `lyrics`.
- For non-English songs, write `lyrics` in the target language.
- `lyrics_optimizer` and `is_instrumental` apply to the generation models, not the cover models.
