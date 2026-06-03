from fastapi.testclient import TestClient

import localinferenceapi as api


def test_backend_serves_tator_ui_and_assets() -> None:
    client = TestClient(api.app)

    root_response = client.get("/")
    assert root_response.status_code == 200
    assert "Tator Annotation Tool" in root_response.text
    assert client.head("/").status_code == 200

    html_response = client.get("/tator.html")
    assert html_response.status_code == 200
    assert "ybat.js" in html_response.text
    assert client.head("/tator.html").status_code == 200

    asset_response = client.get("/ybat.js")
    assert asset_response.status_code == 200
    assert "API_ROOT" in asset_response.text
    assert client.head("/ybat.js").status_code == 200

    legacy_response = client.get("/ybat.html", follow_redirects=False)
    assert legacy_response.status_code == 307
    assert legacy_response.headers["location"] == "/tator.html"
    legacy_head_response = client.head("/ybat.html", follow_redirects=False)
    assert legacy_head_response.status_code == 307
    assert legacy_head_response.headers["location"] == "/tator.html"
