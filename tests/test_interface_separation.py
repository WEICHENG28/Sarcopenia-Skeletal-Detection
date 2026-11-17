from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException

import api_server


STATIC_DIR = Path(api_server.__file__).resolve().parent / "static"


def test_demo_and_clinician_pages_are_separate_no_cache_routes():
    expected = "no-store, no-cache, must-revalidate"
    assert api_server.serve_demo().headers["cache-control"] == expected
    assert api_server.serve_clinician().headers["cache-control"] == expected

    demo = (STATIC_DIR / "demo.html").read_text(encoding="utf-8")
    clinician = (STATIC_DIR / "clinician.html").read_text(encoding="utf-8")
    collection = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    assert "不連接受測者名冊、不啟動攝影機" in demo
    assert "/clinician/records" not in demo
    assert "有肌少症" not in demo
    assert "無肌少症" not in demo
    assert "X-Clinician-Token" in clinician
    assert "/clinician/records" in clinician
    assert "/clinician/review" in clinician
    assert "saveDiagnosis" not in collection
    assert 'action="/label/"' not in collection


def test_clinician_api_fails_closed_without_configured_token(monkeypatch):
    monkeypatch.setattr(api_server, "CLINICIAN_TOKEN", "")
    with pytest.raises(HTTPException) as caught:
        api_server.require_clinician_token("anything")
    assert caught.value.status_code == 503


def test_clinician_api_rejects_wrong_token(monkeypatch):
    monkeypatch.setattr(api_server, "CLINICIAN_TOKEN", "correct-secret")
    with pytest.raises(HTTPException) as caught:
        api_server.require_clinician_token("wrong-secret")
    assert caught.value.status_code == 401
    assert api_server.require_clinician_token("correct-secret") is None


def test_sensitive_routes_declare_clinician_token_dependency():
    protected = {
        ("/clinician/records", "GET"),
        ("/clinician/review", "POST"),
        ("/label/", "POST"),
        ("/result/{user_id}", "GET"),
    }
    found = set()
    for route in api_server.app.routes:
        for method in getattr(route, "methods", set()):
            key = (getattr(route, "path", ""), method)
            if key not in protected:
                continue
            dependency_calls = {
                dependency.call for dependency in route.dependant.dependencies
            }
            assert api_server.require_clinician_token in dependency_calls, key
            found.add(key)
    assert found == protected


def test_server_does_not_enable_wildcard_cors():
    source = Path(api_server.__file__).read_text(encoding="utf-8")
    assert 'allow_origins=["*"]' not in source


def test_clinician_review_round_trip_uses_deidentified_summary(tmp_path, monkeypatch):
    records_path = tmp_path / "records.csv"
    reviews_path = tmp_path / "professional_reviews.json"
    pd.DataFrame([
        {"user_id": "SB01", "action": "Sit", "video_path": "private/video.webm"},
        {"user_id": "SB01", "action": "Walk", "video_path": "private/walk.webm"},
    ]).to_csv(records_path, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(api_server, "RECORDS_CSV", records_path)
    monkeypatch.setattr(api_server, "PROFESSIONAL_REVIEWS_PATH", reviews_path)

    initial = api_server.clinician_records(_authorized=None)["records"]
    assert initial == [{
        "participant_id": "SB01",
        "completed_actions": ["Sit", "Walk"],
        "professional_review": "",
        "review_note": "",
        "reviewed_at": "",
    }]
    assert "video_path" not in initial[0]

    result = api_server.save_clinician_review(
        api_server.ClinicianReviewRequest(
            participant_id="SB01",
            assessment="refer_for_confirmation",
            note="建議安排完整評估",
        ),
        _authorized=None,
    )
    assert result["status"] == "success"
    saved = api_server.clinician_records(_authorized=None)["records"][0]
    assert saved["professional_review"] == "refer_for_confirmation"
    assert saved["review_note"] == "建議安排完整評估"


def test_clinician_review_rejects_unknown_participant(tmp_path, monkeypatch):
    records_path = tmp_path / "records.csv"
    pd.DataFrame([{"user_id": "SB01", "action": "Sit"}]).to_csv(
        records_path, index=False, encoding="utf-8-sig"
    )
    monkeypatch.setattr(api_server, "RECORDS_CSV", records_path)
    monkeypatch.setattr(
        api_server, "PROFESSIONAL_REVIEWS_PATH", tmp_path / "reviews.json"
    )

    with pytest.raises(HTTPException) as caught:
        api_server.save_clinician_review(
            api_server.ClinicianReviewRequest(
                participant_id="SB99",
                assessment="insufficient",
            ),
            _authorized=None,
        )
    assert caught.value.status_code == 404
