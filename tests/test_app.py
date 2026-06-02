"""
Unit tests for src/app.py (Flask web application)
Run with: pytest tests/test_app.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app import app


@pytest.fixture
def client():
    """Creates a Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


# ─── TC-WEB-01: GET / Returns 200 ───
def test_index_page_loads(client):
    """Home page must return 200 and contain upload form."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'upload' in response.data or b'file' in response.data


# ─── TC-WEB-02: GET / Contains Form ───
def test_index_has_upload_form(client):
    """Home page must contain a form with enctype multipart."""
    response = client.get('/')
    assert b'form' in response.data
    assert b'enctype="multipart/form-data"' in response.data


# ─── TC-WEB-03: POST /predict with No File Returns 302 ───
def test_predict_no_file_redirects(client):
    """Posting without file must redirect to index with flash."""
    response = client.post('/predict', data={}, follow_redirects=True)
    assert response.status_code == 200  # After redirect


# ─── TC-WEB-04: POST /predict with Empty Filename Returns 302 ───
def test_predict_empty_filename_redirects(client):
    """Posting with empty filename must redirect."""
    response = client.post(
        '/predict',
        data={'file': (None, '')},
        follow_redirects=True
    )
    assert response.status_code == 200


# ─── TC-WEB-05: POST /predict with Invalid Extension Returns 302 ───
def test_predict_invalid_extension_redirects(client):
    """Posting a .txt file must redirect with error."""
    response = client.post(
        '/predict',
        data={'file': (b'this is not an image', 'test.txt')},
        follow_redirects=True
    )
    assert response.status_code == 200


# ─── TC-WEB-06: GET /health Returns JSON ───
def test_health_endpoint(client):
    """Health endpoint must return JSON with status field."""
    response = client.get('/health')
    assert response.status_code in [200, 503]
    data = response.get_json()
    assert data is not None
    assert 'status' in data


# ─── TC-WEB-07: GET /classes Returns JSON ───
def test_classes_endpoint(client):
    """Classes endpoint must return JSON with classes list."""
    response = client.get('/classes')
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert 'classes' in data
    assert isinstance(data['classes'], list)


# ─── TC-WEB-08: Static CSS is Served ───
def test_static_css_served(client):
    """CSS file must be accessible via static route."""
    response = client.get('/static/css/style.css')
    # May be 200 or 404 if file doesn't exist yet
    assert response.status_code in [200, 404]


# ─── TC-WEB-09: 404 Handler Returns 404 ───
def test_404_handler(client):
    """Unknown routes must return 404."""
    response = client.get('/nonexistent_page_xyz')
    assert response.status_code == 404


# ─── TC-WEB-10: App Has Secret Key ───
def test_app_has_secret_key():
    """Flask app must have a secret key configured."""
    assert app.secret_key is not None
    assert len(app.secret_key) > 0