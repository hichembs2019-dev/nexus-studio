from __future__ import annotations

import base64
import json
import subprocess

import httpx
import pytest

from src.integrations.kling_video_client import (
    BASE_URL,
    KlingBalanceError,
    KlingVideoClient,
    KlingVideoError,
    estimated_price_per_second_usd,
)


def _data_uri(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _fake_ffprobe(monkeypatch, duration: str):
    def fake_run(cmd, *, capture_output, text):
        assert cmd[0] == "ffprobe"
        return subprocess.CompletedProcess(cmd, 0, stdout=duration, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


class _FakeKling:
    """Answers exactly the request shapes in Kling's official docs and
    records every request, so no real (paid) call is ever made."""

    def __init__(self, *, video_billing=None, face_data=None, face_deduction="0.05", lip_deduction="0.5",
                 video_status=("processing", "succeeded"), lip_status=("processing", "succeed"),
                 create_error=None):
        self.requests: list[httpx.Request] = []
        self._video_status = list(video_status)
        self._lip_status = list(lip_status)
        self._video_billing = video_billing if video_billing is not None else [
            {"charge_type": "unit", "amount": "3", "package_type": "video"}
        ]
        self._face_data = face_data if face_data is not None else [
            {"face_id": "0", "face_image": "u", "start_time": 400, "end_time": 5000}
        ]
        self._face_deduction = face_deduction
        self._lip_deduction = lip_deduction
        self._create_error = create_error

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/image-to-video/kling-3.0":
            if self._create_error:
                return httpx.Response(self._create_error[0], json={"code": self._create_error[1], "message": "x"})
            return httpx.Response(200, json={"code": 0, "data": {"id": "T1", "status": "submitted"}})
        if path == "/tasks":
            status = self._video_status.pop(0) if len(self._video_status) > 1 else self._video_status[0]
            task = {"id": "T1", "status": status}
            if status == "succeeded":
                task["outputs"] = [{"type": "video", "id": "V1", "url": "https://cdn.example/v1.mp4", "duration": "5"}]
                task["billing"] = self._video_billing
            if status == "failed":
                task["message"] = "risk control"
            return httpx.Response(200, json={"code": 0, "data": [task]})
        if path == "/v1/videos/identify-face":
            data = {"session_id": "S1", "face_data": self._face_data}
            if self._face_deduction is not None:
                data["final_unit_deduction"] = self._face_deduction
            return httpx.Response(200, json={"code": 0, "data": data})
        if path == "/v1/videos/advanced-lip-sync":
            return httpx.Response(200, json={"code": 0, "data": {"task_id": "L1", "task_status": "submitted"}})
        if path == "/v1/videos/advanced-lip-sync/L1":
            status = self._lip_status.pop(0) if len(self._lip_status) > 1 else self._lip_status[0]
            data = {"task_id": "L1", "task_status": status}
            if status == "succeed":
                data["task_result"] = {"videos": [{"id": "R1", "url": "https://cdn.example/r1.mp4", "duration": "5"}]}
                if self._lip_deduction is not None:
                    data["final_unit_deduction"] = self._lip_deduction
            return httpx.Response(200, json={"code": 0, "data": data})
        if request.url.host == "cdn.example" and path == "/r1.mp4":
            return httpx.Response(200, content=b"lip-synced-mp4")
        return httpx.Response(404, json={"code": 1202})

    def body(self, path: str) -> dict:
        return json.loads(next(r for r in self.requests if r.url.path == path and r.method == "POST").content)


def _client(fake: _FakeKling, **kwargs) -> KlingVideoClient:
    http = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(fake.handler))
    return KlingVideoClient("key-123", http_client=http, sleep=lambda s: None, **kwargs)


def _generate(client: KlingVideoClient) -> bytes:
    return client.generate_speech_to_video(
        prompt="the prince walks toward the throne, camera tracking",
        image_data_uri=_data_uri(b"still-bytes", "image/jpeg"),
        audio_data_uri=_data_uri(b"audio-bytes", "audio/mpeg"),
    )


def test_full_flow_sends_the_documented_requests_and_returns_the_lip_synced_video(monkeypatch):
    _fake_ffprobe(monkeypatch, "4.2")
    fake = _FakeKling()
    client = _client(fake)

    assert _generate(client) == b"lip-synced-mp4"

    assert all(r.headers["Authorization"] == "Bearer key-123" for r in fake.requests if r.url.host != "cdn.example")
    create = fake.body("/image-to-video/kling-3.0")
    assert create["contents"][0] == {"type": "prompt", "text": "the prince walks toward the throne, camera tracking"}
    # Raw Base64, no data: prefix.
    assert create["contents"][1] == {"type": "first_frame", "url": base64.b64encode(b"still-bytes").decode()}
    # 4.2s of audio -> a 5s shot (whole seconds, covers the audio), silent.
    assert create["settings"] == {"resolution": "720p", "duration": 5, "audio": "off", "multi_shot": False}
    assert fake.body("/v1/videos/identify-face") == {"video_id": "V1"}
    lip = fake.body("/v1/videos/advanced-lip-sync")
    assert lip["session_id"] == "S1"
    assert lip["face_choose"] == [{
        "face_id": "0",
        "sound_file": base64.b64encode(b"audio-bytes").decode(),
        "sound_start_time": 0,
        "sound_end_time": 4200,
        # Face appears at 400ms; 400 + 4200 <= 5000, so the audio starts there.
        "sound_insert_time": 400,
    }]


def test_cost_is_read_from_klings_own_deductions(monkeypatch):
    _fake_ffprobe(monkeypatch, "4.2")
    client = _client(_FakeKling(), unit_price_usd=0.1)
    _generate(client)
    # 3 (video) + 0.05 (face) + 0.5 (lip sync) units x $0.1
    assert client.last_cost_usd() == pytest.approx(0.355)


def test_cost_falls_back_to_the_price_table_when_responses_omit_deductions(monkeypatch):
    _fake_ffprobe(monkeypatch, "6.5")
    client = _client(_FakeKling(video_billing=[], face_deduction=None, lip_deduction=None), unit_price_usd=1.0)
    _generate(client)
    # 7s shot x 0.6 + 0.05 + lip sync rounded UP to two 5s blocks (2 x 0.5)
    assert client.last_cost_usd() == pytest.approx(7 * 0.6 + 0.05 + 1.0)


def test_short_audio_is_padded_to_the_three_second_minimum(monkeypatch):
    _fake_ffprobe(monkeypatch, "2.3")
    fake = _FakeKling()
    _generate(_client(fake))
    assert fake.body("/image-to-video/kling-3.0")["settings"]["duration"] == 3


@pytest.mark.parametrize("duration", ["1.9", "15.2"])
def test_audio_outside_klings_limits_fails_before_any_request(monkeypatch, duration):
    _fake_ffprobe(monkeypatch, duration)
    fake = _FakeKling()
    client = _client(fake)
    with pytest.raises(KlingVideoError, match="بدون أي تكلفة"):
        _generate(client)
    assert fake.requests == []
    assert client.last_cost_usd() is None


def test_insert_time_keeps_the_audio_inside_the_video(monkeypatch):
    _fake_ffprobe(monkeypatch, "4.5")
    fake = _FakeKling(face_data=[{"face_id": "2", "start_time": 2000, "end_time": 5000}])
    _generate(_client(fake))
    # 5s shot, 4.5s audio: a start at 2000ms would run past the end.
    assert fake.body("/v1/videos/advanced-lip-sync")["face_choose"][0]["sound_insert_time"] == 500


def test_no_face_found_reports_the_already_paid_video_cost(monkeypatch):
    _fake_ffprobe(monkeypatch, "4.0")
    client = _client(_FakeKling(face_data=[]), unit_price_usd=0.1)
    with pytest.raises(KlingVideoError, match="أي وجه"):
        _generate(client)
    assert client.last_cost_usd() == pytest.approx((3 + 0.05) * 0.1)


def test_failed_video_task_raises_with_klings_reason(monkeypatch):
    _fake_ffprobe(monkeypatch, "4.0")
    with pytest.raises(KlingVideoError, match="risk control"):
        _generate(_client(_FakeKling(video_status=("failed",))))


@pytest.mark.parametrize("code", [1101, 1102])
def test_exhausted_pack_raises_the_balance_error(monkeypatch, code):
    _fake_ffprobe(monkeypatch, "4.0")
    with pytest.raises(KlingBalanceError):
        _generate(_client(_FakeKling(create_error=(429, code))))


def test_estimated_rate_is_motion_plus_lip_sync_per_second():
    assert estimated_price_per_second_usd(1.0, "720p") == pytest.approx(0.6 + 0.1)
    assert estimated_price_per_second_usd(0.14, "1080p") == pytest.approx((0.8 + 0.1) * 0.14)


def test_cost_from_a_previous_clip_never_leaks_into_a_failed_one(monkeypatch):
    fake = _FakeKling()
    client = _client(fake)
    _fake_ffprobe(monkeypatch, "4.0")
    _generate(client)
    assert client.last_cost_usd() is not None
    _fake_ffprobe(monkeypatch, "1.0")
    with pytest.raises(KlingVideoError):
        _generate(client)
    assert client.last_cost_usd() is None
