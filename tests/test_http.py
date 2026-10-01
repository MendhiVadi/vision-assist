import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from vision_assist.api import make_handler


class FakeAnalyzer:
    def analyze(self, data):
        if data == b"bad":
            raise ValueError("Could not decode image")
        return {"output": "a cup ahead", "width": 1, "height": 1, "detections": []}


@pytest.fixture
def base(tmp_path):
    (tmp_path / "index.html").write_text("<h1>hi</h1>")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(FakeAnalyzer(), tmp_path))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def post(url, body):
    req = urllib.request.Request(url, body, {"Content-Type": "image/jpeg"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_and_health(base):
    assert b"<h1>hi</h1>" in urllib.request.urlopen(base + "/").read()
    assert json.loads(urllib.request.urlopen(base + "/health").read()) == {"status": "ok"}


def test_process_ok_and_alias(base):
    assert post(base + "/api/process", b"x")[1]["output"] == "a cup ahead"
    assert post(base + "/analyze", b"x")[0] == 200


def test_bad_image_and_empty(base):
    status, body = post(base + "/api/process", b"bad")
    assert status == 400 and body["output"].startswith("Error")
    assert post(base + "/api/process", b"")[0] == 400
