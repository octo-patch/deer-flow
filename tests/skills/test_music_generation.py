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
    for k in ["MINIMAX_API_KEY", "MINIMAX_API_HOST", "MINIMAX_API_REGION",
              "MINIMAX_MUSIC_MODEL"]:
        monkeypatch.delenv(k, raising=False)


def _post_ok(captured, payload=None):
    def fake_post(url, headers=None, json=None, **kw):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["kwargs"] = kw
        return FakeResp(payload or {"data": {"audio": b"songbytes".hex(), "status": 2},
                                    "base_resp": {"status_code": 0}})
    return fake_post


def _sse(*chunks):
    return [b"data: " + json.dumps(chunk).encode("utf-8") for chunk in chunks]


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
    assert captured["url"] == "https://api.minimaxi.com/v1/music_generation"
    assert captured["headers"]["Authorization"] == "Bearer m"
    assert captured["json"]["model"] == "music-3.0"
    assert captured["json"]["lyrics"] == "[verse]\nla la"
    assert captured["json"]["output_format"] == "hex"
    assert captured["json"]["stream"] is False
    assert captured["json"]["audio_setting"] == {
        "sample_rate": 44100, "bitrate": 256000, "format": "mp3"
    }
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


def test_explicit_optimizer_kept_with_lyrics(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz","lyrics":"[verse]\\nhi","lyrics_optimizer":true}',
                    encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["json"]["lyrics_optimizer"] is True
    assert captured["json"]["lyrics"] == "[verse]\nhi"


def test_model_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_MUSIC_MODEL", "music-2.6")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz","lyrics":"[verse]\\nhi"}', encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["json"]["model"] == "music-2.6"


def test_spec_model_wins_over_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_MUSIC_MODEL", "music-2.6")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"jazz","model":"music-3.0-free","lyrics":"[verse]\\nhi"}',
                    encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["json"]["model"] == "music-3.0-free"


def test_global_region_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_API_REGION", "global_en")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x"}', encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["url"] == "https://api.minimax.io/v1/music_generation"


def test_host_override_wins_over_region(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_API_REGION", "global")
    monkeypatch.setenv("MINIMAX_API_HOST", "https://gateway.example.com/")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x"}', encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["url"] == "https://gateway.example.com/v1/music_generation"


def test_unknown_region_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_API_REGION", "mars")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown MiniMax region"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_region_follows_a_known_host_override(monkeypatch, tmp_path):
    """A global host without an explicit region must not carry the China-only field."""
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_API_HOST", "https://api.minimax.io")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x","aigc_watermark":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="aigc_watermark"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_url_output_is_downloaded(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(
        captured,
        payload={"data": {"audio": "https://files.example.com/song.wav", "status": 2},
                 "base_resp": {"status_code": 0}},
    ))

    def fake_get(url, **kw):
        captured["download"] = url
        return FakeResp(content=b"wavbytes")

    monkeypatch.setattr(mus.requests, "get", fake_get)
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x","output_format":"url",'
                    '"audio_setting":{"format":"wav","sample_rate":24000,"bitrate":128000}}',
                    encoding="utf-8")
    out = tmp_path / "o.wav"
    mus.generate_music(str(spec), str(out))
    assert captured["json"]["output_format"] == "url"
    assert captured["json"]["audio_setting"]["format"] == "wav"
    assert captured["download"] == "https://files.example.com/song.wav"
    assert out.read_bytes() == b"wavbytes"


def test_pcm_format_accepted(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x","audio_setting":{"format":"pcm","sample_rate":16000,'
                    '"bitrate":32000}}', encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.pcm"))
    assert captured["json"]["audio_setting"] == {
        "sample_rate": 16000, "bitrate": 32000, "format": "pcm"
    }


@pytest.mark.parametrize("audio_setting", [
    '{"format":"flac"}',
    '{"sample_rate":48000}',
    '{"bitrate":320000}',
    '{"channel":1}',
])
def test_invalid_audio_setting_raises(monkeypatch, tmp_path, audio_setting):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")

    def fake_post(url, headers=None, json=None, **kw):  # pragma: no cover
        raise AssertionError("must not call the API with an invalid audio_setting")

    monkeypatch.setattr(mus.requests, "post", fake_post)
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x","audio_setting":' + audio_setting + "}", encoding="utf-8")
    with pytest.raises(ValueError, match="audio_setting"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_invalid_output_format_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x","output_format":"base64"}', encoding="utf-8")
    with pytest.raises(ValueError, match="output_format"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_stream_rejects_url_output(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x","stream":true,"output_format":"url"}', encoding="utf-8")
    with pytest.raises(ValueError, match="output_format"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_stream_concatenates_hex_fragments(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}

    def fake_post(url, headers=None, json=None, **kw):
        captured["json"] = json
        captured["stream"] = kw.get("stream")
        return FakeResp(lines=_sse(
            {"data": {"audio": b"song".hex(), "status": 1}, "base_resp": {"status_code": 0}},
            {"data": {"audio": b"bytes".hex(), "status": 1}, "base_resp": {"status_code": 0}},
            {"data": {"status": 2}, "base_resp": {"status_code": 0}},
        ) + [b"", b"data: [DONE]"])

    monkeypatch.setattr(mus.requests, "post", fake_post)
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x","stream":true}', encoding="utf-8")
    out = tmp_path / "o.mp3"
    mus.generate_music(str(spec), str(out))
    assert captured["json"]["stream"] is True
    assert captured["stream"] is True
    assert out.read_bytes() == b"songbytes"


def test_stream_raises_when_never_completed(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")

    def fake_post(url, headers=None, json=None, **kw):
        return FakeResp(lines=_sse(
            {"data": {"audio": b"song".hex(), "status": 1}, "base_resp": {"status_code": 0}},
        ))

    monkeypatch.setattr(mus.requests, "post", fake_post)
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x","stream":true}', encoding="utf-8")
    with pytest.raises(Exception, match="still in progress"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_stream_raises_on_base_resp_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")

    def fake_post(url, headers=None, json=None, **kw):
        return FakeResp(lines=_sse(
            {"base_resp": {"status_code": 1002, "status_msg": "rate limited"}},
        ))

    monkeypatch.setattr(mus.requests, "post", fake_post)
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x","stream":true}', encoding="utf-8")
    with pytest.raises(Exception, match="1002"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_in_progress_status_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setattr(mus.requests, "post", _post_ok(
        {}, payload={"data": {"audio": b"partial".hex(), "status": 1},
                     "base_resp": {"status_code": 0}},
    ))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x"}', encoding="utf-8")
    with pytest.raises(Exception, match="still in progress"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_unexpected_status_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setattr(mus.requests, "post", _post_ok(
        {}, payload={"data": {"audio": b"x".hex(), "status": 7},
                     "base_resp": {"status_code": 0}},
    ))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x"}', encoding="utf-8")
    with pytest.raises(Exception, match="unexpected data.status"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_watermark_sent_on_cn_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x","aigc_watermark":true}', encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["json"]["aigc_watermark"] is True


def test_watermark_rejected_on_global_endpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setenv("MINIMAX_API_REGION", "global_en")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x","aigc_watermark":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="aigc_watermark"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_watermark_rejected_while_streaming(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x","stream":true,"aigc_watermark":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-streaming"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_cover_model_sends_reference_url(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    spec = tmp_path / "s.json"
    spec.write_text('{"model":"music-cover","prompt":"acoustic cover, warm",'
                    '"audio_url":"https://files.example.com/ref.mp3"}', encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["json"]["model"] == "music-cover"
    assert captured["json"]["audio_url"] == "https://files.example.com/ref.mp3"
    assert "lyrics_optimizer" not in captured["json"]
    assert "is_instrumental" not in captured["json"]


def test_cover_model_encodes_local_audio_file(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    captured = {}
    monkeypatch.setattr(mus.requests, "post", _post_ok(captured))
    reference = tmp_path / "ref.mp3"
    reference.write_bytes(b"reference-audio")
    spec = tmp_path / "s.json"
    spec.write_text(json.dumps({"model": "music-cover-free", "prompt": "lofi cover",
                                "audio_file": str(reference)}), encoding="utf-8")
    mus.generate_music(str(spec), str(tmp_path / "o.mp3"))
    assert captured["json"]["audio_base64"] == base64.b64encode(b"reference-audio").decode()
    assert "audio_file" not in captured["json"]


def test_cover_model_rejects_two_reference_inputs(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    spec = tmp_path / "s.json"
    spec.write_text('{"model":"music-cover","prompt":"cover","audio_url":"https://a/b.mp3",'
                    '"cover_feature_id":"feat-1"}', encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one reference input"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_cover_model_requires_a_reference_input(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    spec = tmp_path / "s.json"
    spec.write_text('{"model":"music-cover","prompt":"cover"}', encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one reference input"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_cover_feature_id_requires_lyrics(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    spec = tmp_path / "s.json"
    spec.write_text('{"model":"music-cover","prompt":"cover","cover_feature_id":"feat-1"}',
                    encoding="utf-8")
    with pytest.raises(ValueError, match="lyrics"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_reference_input_rejected_for_generation_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    spec = tmp_path / "s.json"
    spec.write_text('{"prompt":"x","audio_url":"https://files.example.com/ref.mp3"}',
                    encoding="utf-8")
    with pytest.raises(ValueError, match="requires a cover model"):
        mus.generate_music(str(spec), str(tmp_path / "o.mp3"))


def test_oversized_reference_audio_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "m")
    monkeypatch.setattr(mus, "COVER_INPUT_MAX_BYTES", 8)
    reference = tmp_path / "ref.mp3"
    reference.write_bytes(b"way too much audio")
    spec = tmp_path / "s.json"
    spec.write_text(json.dumps({"model": "music-cover", "prompt": "cover",
                                "audio_file": str(reference)}), encoding="utf-8")
    with pytest.raises(ValueError, match="Reference audio"):
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
