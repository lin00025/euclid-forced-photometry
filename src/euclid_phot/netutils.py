"""Network helpers shared by every band's data access.

IRSA TAP/SIA, S3 byte-range reads, unwise.me and the Spitzer archives
sometimes fail transiently, so every network call goes through ``retry``:
a few tries with a short wait in between. ``download_file`` writes a file
atomically over verified HTTPS.
"""
from __future__ import annotations

import os
import hashlib
import http.client
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

# OSError covers socket/timeout errors (s3fs, urllib); pyvo/astroquery raise
# requests exceptions, which subclass OSError in requests>=2.
_DEFAULT_EXCEPTIONS = (OSError, TimeoutError)

# anon=True is required to read the public IRSA S3 bucket. s3fs fetches whole
# blocks of 50 MiB by default, so a header or a small section read would pull
# 50 MB each; 1 MiB keeps byte-range reads close to what is actually needed.
S3_FSSPEC_KWARGS = {"anon": True, "default_block_size": 1 << 20}


def retry(fn, *, attempts=3, base_delay=2.0,
          exceptions=_DEFAULT_EXCEPTIONS, what=""):
    """Call ``fn()`` up to ``attempts`` times, waiting between tries.

    ``fn`` is a zero-argument callable (close over its arguments). Only
    ``exceptions`` are retried; a ``ValueError`` (bad request, missing
    product) fails immediately unless listed. The last failure re-raises.
    """
    for k in range(attempts):
        try:
            return fn()
        except exceptions as exc:
            if k == attempts - 1:
                raise
            print(f"  {what or 'network call'} failed ({exc!r}); "
                  f"retry {k + 1}/{attempts - 1} in {base_delay:.0f}s")
            time.sleep(base_delay)


def ensure_https_ca_bundle() -> None:
    """Use certifi only when Python has no usable configured CA store."""
    configured = os.environ.get("SSL_CERT_FILE")
    if configured and Path(configured).is_file():
        return
    paths = ssl.get_default_verify_paths()
    if ((paths.cafile and Path(paths.cafile).is_file())
            or (paths.capath and Path(paths.capath).is_dir())):
        return
    try:
        import certifi
    except ImportError:
        return
    bundle = certifi.where()
    if Path(bundle).is_file():
        os.environ["SSL_CERT_FILE"] = bundle


def https_ssl_context() -> ssl.SSLContext:
    """Return a verified context that also works with aiohttp/fsspec.

    Some Python installations do not have a usable OpenSSL CA path.  Setting
    ``SSL_CERT_FILE`` is enough for urllib, but aiohttp may create its default
    context before that environment variable is set.  Supplying this context
    explicitly keeps remote FITS range reads verified and deterministic.
    """
    ensure_https_ca_bundle()
    configured = os.environ.get("SSL_CERT_FILE")
    if configured and Path(configured).is_file():
        return ssl.create_default_context(cafile=configured)
    return ssl.create_default_context()


def download_file(url: str, path: str | Path, *, force: bool = False,
                  timeout: float = 300.0, attempts: int = 4) -> Path:
    """Download atomically, retrying stalled reads and resuming partial files.

    A failed download stays in ``<path>.tmp`` for the next attempt or run.
    The sidecar records the URL hash so bytes from a different archive
    product can never be appended to it. The final path appears only after
    the response completes and any reported content length is verified.
    """
    path = Path(path)
    if path.exists() and not force:
        return path
    if attempts < 1:
        raise ValueError("attempts must be at least one")
    ensure_https_ca_bundle()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    marker = path.with_name(path.name + ".tmp.urlsha256")
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    if temporary.exists() and (not marker.exists()
                               or marker.read_text() != url_hash):
        temporary.unlink()
    marker.write_text(url_hash)

    for attempt in range(attempts):
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", None) or response.getcode()
                content_range = response.headers.get("Content-Range", "")
                if offset and status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)",
                                         content_range)
                    if match is None or int(match.group(1)) != offset:
                        temporary.unlink(missing_ok=True)
                        raise OSError("archive returned an invalid resume range")
                    total_size = None if match.group(3) == "*" else int(match.group(3))
                elif offset and status == 200:
                    # This server ignored Range; safely restart from byte zero.
                    offset = 0
                    total_size = None
                elif offset:
                    temporary.unlink(missing_ok=True)
                    raise OSError(f"archive returned unexpected HTTP {status} for resume")
                else:
                    if status != 200:
                        raise OSError(f"archive returned unexpected HTTP {status}")
                    total_size = None

                expected = response.headers.get("Content-Length")
                expected = int(expected) if expected is not None else None
                received = 0
                with temporary.open("ab" if offset else "wb") as output:
                    while chunk := response.read(1 << 20):
                        output.write(chunk)
                        received += len(chunk)
                if expected is not None and received != expected:
                    raise OSError(f"incomplete download: {received}/{expected} bytes")
                if total_size is not None and temporary.stat().st_size != total_size:
                    raise OSError("resumed download ended before the expected file size")
            temporary.replace(path)
            marker.unlink(missing_ok=True)
            return path
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and offset:
                # A timeout can occur after the final byte but before EOF.
                # The server then rejects the next start=filesize request.
                match = re.fullmatch(r"bytes \*/(\d+)",
                                     exc.headers.get("Content-Range", ""))
                if match is not None and int(match.group(1)) == offset:
                    temporary.replace(path)
                    marker.unlink(missing_ok=True)
                    return path
                temporary.unlink(missing_ok=True)
            if attempt == attempts - 1:
                raise
            print(f"  download {path.name} failed ({exc!r}); "
                  f"retry {attempt + 1}/{attempts - 1}")
        except (OSError, http.client.IncompleteRead) as exc:
            if attempt == attempts - 1:
                raise
            print(f"  download {path.name} failed ({exc!r}); "
                  f"retry {attempt + 1}/{attempts - 1}")
        time.sleep(min(2 ** (attempt + 1), 30))
    raise RuntimeError("unreachable download retry state")


_BITPIX_DTYPE = {8: "u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}


def read_s3_image_section(hdu, slices, s3_path):
    """Pixels of ``hdu[slices]``; for an S3-hosted mosaic, one concurrent
    byte-range request per row.

    A cutout of N rows is N short strips spaced one mosaic row apart.
    Reading them through the file object pulls the whole span between
    them, and sequential requests pay the S3 round trip N times. Fetching
    the strips together with ``cat_ranges`` transfers only the cutout and
    overlaps the requests. Scaled, blanked or compressed images fall back
    to the ordinary section read.
    """
    header = hdu.header
    scaled = (header.get("BSCALE", 1) != 1 or header.get("BZERO", 0) != 0
              or "BLANK" in header)
    dtype = _BITPIX_DTYPE.get(header.get("BITPIX"))
    offset = getattr(hdu, "_data_offset", None)
    if (s3_path is None or scaled or dtype is None or offset is None
            or header.get("NAXIS") != 2 or hdu.is_image is False
            or type(hdu).__name__ == "CompImageHDU"):
        return np.asarray(hdu.section[slices]).copy()

    import s3fs
    rows, cols = (range(*sl.indices(n)) for sl, n in zip(slices, hdu.shape))
    nx = int(header["NAXIS1"])
    bpp = np.dtype(dtype).itemsize
    starts = [offset + (r * nx + cols.start) * bpp for r in rows]
    ends = [start + len(cols) * bpp for start in starts]
    fs = s3fs.S3FileSystem(**S3_FSSPEC_KWARGS)
    chunks = retry(lambda: fs.cat_ranges([s3_path] * len(starts), starts, ends,
                                         on_error="raise"),
                   what=f"S3 cutout rows {s3_path.split('/')[-1][:40]}")
    data = np.frombuffer(b"".join(chunks), dtype=dtype)
    return data.reshape(len(rows), len(cols)).astype(data.dtype.newbyteorder("="))
