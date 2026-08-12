import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skill_loader import FakeResp, load  # noqa: E402

mus = load("music-generation")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ["MINIMAX_API_KEY", "MINIMAX_API_HOST", "MINIMAX_MUSIC_MODEL"]:
        monkeypatch.delenv(k, raising=False)


def _post_ok(captured):
    def fake_post(url, headers=None, json=None, **kw):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResp(
            {
                "data": {"audio": b"songbytes".hex(), "status": 2},
                "base_resp": {"status_code": 0},
            }
        )

    return fake_post


def test_with_lyrics_payload_and_writes(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text(
        '{"title":"X","prompt":"pop, happy","lyrics":"[verse]\\nla la",'
        '"aigc_watermark":true}',
        encoding="utf-8",
    )
    out = tmp_path / "o.mp3"
    msg = mus.generate_music(str(spec), str(out))
    assert out.read_bytes() == b"songbytes"
    assert captured["url"] == "https://api.minimaxi.com/v1/music_generation"
    assert captured["headers"]["Authorization"] == "Bearer m"
    assert captured["json"]["model"] == "music-3.0"
    assert captured["json"]["lyrics"] == "[verse]\nla la"
    assert captured["json"]["aigc_watermark"] is True
    assert captured["json"]["output_format"] == "hex"
    assert captured["json"]["stream"] is False
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


@pytest.mark.parametrize(
    ("audio_field", "audio_value"),
    [
        ("audio_url", "https://example.com/input.mp3"),
        ("audio_base64", "c291cmNlLWF1ZGlv"),
    ],
)
def test_spec_model_and_audio_inputs(monkeypatch, tmp_path, audio_field, audio_value):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text(
        json.dumps(
            {
                "model": "music-3.0-free",
                audio_field: audio_value,
                "cover_feature_id": "feature-1",
                "is_instrumental": True,
            }
        ),
        encoding="utf-8",
    )
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["json"]["model"] == "music-3.0-free"
    assert captured["json"][audio_field] == audio_value
    assert captured["json"]["cover_feature_id"] == "feature-1"
    assert "prompt" not in captured["json"]


def test_global_url_response_and_wav_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_API_HOST", "https://api.minimax.io")
    captured = {}

    def fake_post(url, headers=None, json=None, **kw):
        captured["url"] = url
        captured["json"] = json
        return FakeResp(
            {
                "data": {"audio": "https://example.com/song.wav", "status": 2},
                "base_resp": {"status_code": 0},
            }
        )

    monkeypatch.setattr(mus.requests, "post", fake_post)
    monkeypatch.setattr(
        mus.requests, "get", lambda *args, **kwargs: FakeResp(content=b"wav")
    )
    spec = tmp_path / "s.json"
    spec.write_text(
        '{"prompt":"jazz","output_format":"url",'
        '"audio_setting":{"sample_rate":32000,"bitrate":128000,"format":"wav"}}',
        encoding="utf-8",
    )
    out = tmp_path / "o.wav"
    mus.generate_music(str(spec), str(out))
    assert captured["url"] == "https://api.minimax.io/v1/music_generation"
    assert captured["json"]["output_format"] == "url"
    assert captured["json"]["audio_setting"] == {
        "sample_rate": 32000,
        "bitrate": 128000,
        "format": "wav",
    }
    assert out.read_bytes() == b"wav"


def test_streaming_hex_chunks(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}

    class StreamResp(FakeResp):
        def iter_lines(self):
            yield b'data: {"data":{"audio":"6f6e65","status":1},"base_resp":{"status_code":0}}'
            yield b'data: {"data":{"audio":"74776f","status":2},"base_resp":{"status_code":0}}'
            yield b"data: [DONE]"

    def fake_post(url, headers=None, json=None, **kw):
        captured["json"] = json
        captured["stream"] = kw["stream"]
        return StreamResp()

    monkeypatch.setattr(mus.requests, "post", fake_post)
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"ambient","stream":true}', encoding="utf-8")
    out = tmp_path / "o.mp3"
    mus.generate_music(str(spec), str(out))
    assert captured["stream"] is True
    assert captured["json"]["output_format"] == "hex"
    assert out.read_bytes() == b"onetwo"


def test_streaming_requires_completed_status(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")

    class StreamResp(FakeResp):
        def iter_lines(self):
            yield b'data: {"data":{"audio":"6f6e65","status":1},"base_resp":{"status_code":0}}'

    monkeypatch.setattr(mus.requests, "post", lambda *args, **kwargs: StreamResp())
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"ambient","stream":true}', encoding="utf-8")
    with pytest.raises(Exception, match="did not complete"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_raises_on_base_resp_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")

    def fake_post(url, headers=None, json=None, **kw):
        return FakeResp(
            {"base_resp": {"status_code": 1008, "status_msg": "no balance"}}
        )

    monkeypatch.setattr(mus.requests, "post", fake_post)
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x","lyrics":"[verse]\\ny"}', encoding="utf-8")
    with pytest.raises(Exception) as e:
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert "1008" in str(e.value)


def test_requires_base_response_status_code(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setattr(
        mus.requests,
        "post",
        lambda *args, **kwargs: FakeResp(
            {"data": {"audio": b"song".hex(), "status": 2}, "base_resp": {}}
        ),
    )
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x"}', encoding="utf-8")
    with pytest.raises(Exception, match="MiniMax error None"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_missing_api_key_returns_message(monkeypatch, tmp_path):
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x"}', encoding="utf-8")
    msg = mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert "MINIMAX_API_KEY" in msg


def test_raises_on_missing_audio_data(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")

    def fake_post(url, headers=None, json=None, **kw):
        return FakeResp({"data": {"status": 2}, "base_resp": {"status_code": 0}})

    monkeypatch.setattr(mus.requests, "post", fake_post)
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x"}', encoding="utf-8")
    with pytest.raises(Exception, match="no audio data"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_raises_on_incomplete_status(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")

    def fake_post(url, headers=None, json=None, **kw):
        return FakeResp(
            {
                "data": {"audio": b"partial".hex(), "status": 1},
                "base_resp": {"status_code": 0},
            }
        )

    monkeypatch.setattr(mus.requests, "post", fake_post)
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x"}', encoding="utf-8")
    with pytest.raises(Exception, match="did not complete"):
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
