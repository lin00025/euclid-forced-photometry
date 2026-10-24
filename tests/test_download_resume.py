"""A timed-out archive transfer must not force a second full download."""

import urllib.request
import urllib.error

import pytest

from euclid_phot.netutils import download_file


class _Response:
    def __init__(self, data, *, status, headers, fail_after=None):
        self.data = data
        self.status = status
        self.headers = headers
        self.fail_after = fail_after
        self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            raise TimeoutError("archive stalled")
        if self.reads == 1:
            return self.data
        return b""


def test_download_resumes_after_timeout_on_next_run(tmp_path, monkeypatch):
    url = "https://archive.example/test.fits"
    path = tmp_path / "uncertainty.fits"
    requests = []

    def open_url(request, timeout):
        assert isinstance(request, urllib.request.Request)
        assert timeout == 300.0
        requests.append(request.get_header("Range"))
        if len(requests) == 1:
            return _Response(b"abc", status=200,
                             headers={"Content-Length": "6"}, fail_after=1)
        return _Response(b"def", status=206,
                         headers={"Content-Length": "3",
                                  "Content-Range": "bytes 3-5/6"})

    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    with pytest.raises(TimeoutError, match="archive stalled"):
        download_file(url, path, attempts=1)
    assert not path.exists()
    assert path.with_name(path.name + ".tmp").read_bytes() == b"abc"

    assert download_file(url, path, attempts=1) == path
    assert path.read_bytes() == b"abcdef"
    assert requests == [None, "bytes=3-"]
    assert not path.with_name(path.name + ".tmp").exists()
    assert not path.with_name(path.name + ".tmp.urlsha256").exists()


def test_download_restarts_if_server_ignores_range(tmp_path, monkeypatch):
    url = "https://archive.example/test.fits"
    path = tmp_path / "science.fits"
    requests = []

    def open_url(request, timeout):
        requests.append(request.get_header("Range"))
        if len(requests) == 1:
            return _Response(b"abc", status=200,
                             headers={"Content-Length": "6"}, fail_after=1)
        return _Response(b"abcdef", status=200,
                         headers={"Content-Length": "6"})

    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    with pytest.raises(TimeoutError):
        download_file(url, path, attempts=1)
    download_file(url, path, attempts=1)
    assert path.read_bytes() == b"abcdef"
    assert requests == [None, "bytes=3-"]


def test_download_accepts_complete_temp_after_timeout_at_eof(tmp_path, monkeypatch):
    path = tmp_path / "coverage.fits"
    calls = 0

    def open_url(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _Response(b"abcdef", status=200,
                             headers={"Content-Length": "6"}, fail_after=1)
        assert request.get_header("Range") == "bytes=6-"
        raise urllib.error.HTTPError(
            request.full_url, 416, "Range Not Satisfiable",
            {"Content-Range": "bytes */6"}, None)

    monkeypatch.setattr(urllib.request, "urlopen", open_url)
    with pytest.raises(TimeoutError):
        download_file("https://archive.example/test.fits", path, attempts=1)
    assert download_file("https://archive.example/test.fits", path, attempts=1) == path
    assert path.read_bytes() == b"abcdef"
