import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skill_loader import FakeResp, load  # noqa: E402

mus = load("music-generation")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ["MINIMAX_API_KEY", "MINIMAX_API_HOST", "MINIMAX_MUSIC_MODEL",
              "MINIMAX_MUSIC_REGION"]:
        monkeypatch.delenv(k, raising=False)


class FakeStreamResp(FakeResp):
    """FakeResp that also replays server-sent event lines."""

    def __init__(self, lines, json_data=None, status_code=200):
        super().__init__(json_data=json_data, status_code=status_code)
        self._lines = lines

    def iter_lines(self):
        yield from self._lines


def _post_ok(captured, payload=None):
    def fake_post(url, headers=None, json=None, **kw):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["kwargs"] = kw
        return FakeResp(payload or {"data": {"audio": b"songbytes".hex(), "status": 2},
                                    "base_resp": {"status_code": 0}})
    return fake_post


def test_with_lyrics_payload_and_writes(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"title":"X","prompt":"pop, happy","lyrics":"[verse]\\nla la"}',
                    encoding="utf-8")
    out = tmp_path / "o.mp3"
    msg = mus.generate_music(str(spec), str(out))
    assert out.read_bytes() == b"songbytes"
    assert captured["url"].endswith("/v1/music_generation")
    assert captured["headers"]["Authorization"] == "Bearer m"
    assert captured["json"]["model"] == "music-3.0"
    assert captured["json"]["lyrics"] == "[verse]\nla la"
    assert captured["json"]["output_format"] == "hex"
    assert captured["json"]["stream"] is False
    assert captured["json"]["audio_setting"]["format"] == "mp3"
    assert "Successfully generated music" in msg


def test_instrumental_sets_flag(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"lofi beats","is_instrumental":true}', encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["json"]["is_instrumental"] is True
    assert "lyrics" not in captured["json"]
    assert "lyrics_optimizer" not in captured["json"]


def test_no_lyrics_uses_optimizer(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"sad ballad"}', encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["json"]["lyrics_optimizer"] is True
    assert "lyrics" not in captured["json"]


def test_model_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_MUSIC_MODEL", "music-2.6")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz","lyrics":"[verse]\\nhi"}', encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["json"]["model"] == "music-2.6"


def test_unknown_model_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_MUSIC_MODEL", "music-9.9")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown music model"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_default_region_uses_china_host(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz"}', encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["url"] == "https://api.minimaxi.com/v1/music_generation"


def test_global_region_uses_global_host(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_MUSIC_REGION", "GLOBAL")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz"}', encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["url"] == "https://api.minimax.io/v1/music_generation"


def test_api_host_overrides_region(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_MUSIC_REGION", "global")
    monkeypatch.setenv("MINIMAX_API_HOST", "https://proxy.example/")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz"}', encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["url"] == "https://proxy.example/v1/music_generation"


def test_unknown_region_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_MUSIC_REGION", "mars")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz"}', encoding="utf-8")
    with pytest.raises(ValueError, match="MINIMAX_MUSIC_REGION"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_audio_setting_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz","audio_setting":{"format":"WAV","sample_rate":24000}}',
                    encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.wav"))
    assert captured["json"]["audio_setting"]["format"] == "wav"
    assert captured["json"]["audio_setting"]["sample_rate"] == 24000
    assert captured["json"]["audio_setting"]["bitrate"] == 256000


def test_pcm_audio_format_allowed(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz","audio_setting":{"format":"pcm"}}', encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.pcm"))
    assert captured["json"]["audio_setting"]["format"] == "pcm"


def test_unsupported_audio_format_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz","audio_setting":{"format":"ogg"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="audio_setting.format"):
        mus.generate_music(str(spec), str(tmp_path / "o.ogg"))


def test_url_output_downloads_the_link(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(
        captured, {"data": {"audio": "https://files.example/song.mp3", "status": 2},
                   "base_resp": {"status_code": 0}}))
    downloaded = {}

    def fake_get(url, **kw):
        downloaded["url"] = url
        return FakeResp(content=b"downloadedsong")

    monkeypatch.setattr(mus.requests, "get", fake_get)
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz","output_format":"url"}', encoding="utf-8")
    out = tmp_path / "o.mp3"
    mus.generate_music(str(spec), str(out))
    assert captured["json"]["output_format"] == "url"
    assert downloaded["url"] == "https://files.example/song.mp3"
    assert out.read_bytes() == b"downloadedsong"


def test_unknown_output_format_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz","output_format":"flac"}', encoding="utf-8")
    with pytest.raises(ValueError, match="output_format"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_stream_joins_hex_chunks(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    lines = [
        b"",
        b"data: " + json.dumps({"data": {"audio": b"song".hex(), "status": 1},
                                "base_resp": {"status_code": 0}}).encode(),
        b"data: " + json.dumps({"data": {"audio": b"bytes".hex(), "status": 2},
                                "base_resp": {"status_code": 0}}).encode(),
        b"data: [DONE]",
    ]

    def fake_post(url, headers=None, json=None, **kw):
        captured["json"] = json
        captured["kwargs"] = kw
        return FakeStreamResp(lines)

    monkeypatch.setattr(mus.requests, "post", fake_post)
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz","stream":true}', encoding="utf-8")
    out = tmp_path / "o.mp3"
    mus.generate_music(str(spec), str(out))
    assert captured["json"]["stream"] is True
    assert captured["kwargs"]["stream"] is True
    assert out.read_bytes() == b"songbytes"


def test_stream_requires_hex_output(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz","stream":true,"output_format":"url"}',
                    encoding="utf-8")
    with pytest.raises(ValueError, match="hex"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_stream_without_completion_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    lines = [b"data: " + json.dumps({"data": {"audio": b"song".hex(), "status": 1},
                                     "base_resp": {"status_code": 0}}).encode()]
    monkeypatch.setattr(mus.requests, "post",
                        lambda url, headers=None, json=None, **kw: FakeStreamResp(lines))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz","stream":true}', encoding="utf-8")
    with pytest.raises(Exception, match="before data.status reported completion"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_stream_surfaces_base_resp_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    lines = [b"data: " + json.dumps(
        {"base_resp": {"status_code": 1002, "status_msg": "rate limited"}}).encode()]
    monkeypatch.setattr(mus.requests, "post",
                        lambda url, headers=None, json=None, **kw: FakeStreamResp(lines))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz","stream":true}', encoding="utf-8")
    with pytest.raises(Exception) as e:
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert "1002" in str(e.value)


def test_in_progress_status_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(
        captured, {"data": {"audio": b"partial".hex(), "status": 1},
                   "base_resp": {"status_code": 0}}))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz"}', encoding="utf-8")
    with pytest.raises(Exception, match="still in progress"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_watermark_allowed_on_china_region(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz","aigc_watermark":true}', encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["json"]["aigc_watermark"] is True


def test_watermark_rejected_on_global_region(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_MUSIC_REGION", "global")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz","aigc_watermark":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="aigc_watermark"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_watermark_rejected_when_streaming(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz","stream":true,"aigc_watermark":true}',
                    encoding="utf-8")
    with pytest.raises(ValueError, match="stream"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_cover_model_accepts_audio_url(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_MUSIC_MODEL", "music-cover")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"warm acoustic cover","audio_url":"https://a/b.mp3"}',
                    encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["json"]["model"] == "music-cover"
    assert captured["json"]["audio_url"] == "https://a/b.mp3"
    # Cover lyrics come from the reference audio, so no lyrics flags are sent.
    assert "lyrics_optimizer" not in captured["json"]
    assert "lyrics" not in captured["json"]


def test_cover_model_requires_exactly_one_reference(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_MUSIC_MODEL", "music-cover-free")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"cover","audio_url":"https://a/b.mp3",'
                    '"cover_feature_id":"feat-1"}', encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_cover_model_without_reference_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_MUSIC_MODEL", "music-cover")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"cover"}', encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_cover_feature_id_requires_lyrics(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_MUSIC_MODEL", "music-cover")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"cover","cover_feature_id":"feat-1"}', encoding="utf-8")
    with pytest.raises(ValueError, match="lyrics"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_cover_feature_id_with_lyrics_is_sent(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_MUSIC_MODEL", "music-cover")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"cover","cover_feature_id":"feat-1",'
                    '"lyrics":"[verse]\\nhold the line"}', encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["json"]["cover_feature_id"] == "feat-1"
    assert captured["json"]["lyrics"] == "[verse]\nhold the line"


def test_cover_accepts_inline_base64_audio(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_MUSIC_MODEL", "music-cover")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    encoded = base64.b64encode(b"referenceaudio").decode()
    spec = tmp_path / "s.json"
    spec.write_text(json.dumps({"prompt": "cover", "audio_base64": encoded}),
                    encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["json"]["audio_base64"] == encoded


def test_cover_base64_over_size_limit_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_MUSIC_MODEL", "music-cover")
    monkeypatch.setattr(mus, "COVER_INPUT_MAX_MB", 0)  # keep the fixture small
    spec = tmp_path / "s.json"
    spec.write_text(json.dumps({"prompt": "cover",
                                "audio_base64": base64.b64encode(b"x" * 4096).decode()}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="reference audio limit"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_cover_fields_rejected_for_generation_models(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz","audio_url":"https://a/b.mp3"}', encoding="utf-8")
    with pytest.raises(ValueError, match="cover"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_raises_on_base_resp_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")

    def fake_post(url, headers=None, json=None, **kw):
        return FakeResp({"base_resp": {"status_code": 1008, "status_msg": "no balance"}})

    monkeypatch.setattr(mus.requests, "post", fake_post)
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x","lyrics":"[verse]\\ny"}', encoding="utf-8")
    with pytest.raises(Exception) as e:
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert "1008" in str(e.value)


def test_missing_api_key_returns_message(monkeypatch, tmp_path):
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x"}', encoding="utf-8")
    msg = mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert "MINIMAX_API_KEY" in msg


def test_raises_on_missing_audio_data(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")

    def fake_post(url, headers=None, json=None, **kw):
        return FakeResp({"base_resp": {"status_code": 0}})  # no "data" key

    monkeypatch.setattr(mus.requests, "post", fake_post)
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x"}', encoding="utf-8")
    with pytest.raises(Exception, match="no audio data"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_empty_prompt_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")

    def fake_post(url, headers=None, json=None, **kw):  # pragma: no cover
        raise AssertionError("must not call the API when prompt is missing")

    monkeypatch.setattr(mus.requests, "post", fake_post)
    spec = tmp_path / "s.json"
    spec.write_text('{"title":"X","lyrics":"[verse]\\nhi"}', encoding="utf-8")  # no prompt
    with pytest.raises(ValueError, match="prompt"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_empty_lyrics_falls_back_to_optimizer(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x","lyrics":""}', encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["json"]["lyrics_optimizer"] is True
    assert "lyrics" not in captured["json"]
