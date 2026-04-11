"""PDF / full-text fetch pipeline.

Waterfall:
  SKIP  — non-document extensions (.bib, .avi, ...)
  API   — Zenodo REST API (bypasses bot detection)
  L1    — httpx: direct PDF or HTML→PDF link extraction
            * citation_pdf_url meta  (Springer/Nature/Wiley/PLOS/BMC/...)
            * <link rel="alternate" type="application/pdf">
            * <a href="*.pdf">
            * <a class/id containing "pdf">
  L2    — Playwright + stealth (JS-rendered pages, bot detection)
  L3    — HTML text fallback (validates not error/bot page)
  FAIL  — all layers failed
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx
from lxml import html as lxml_html

CHROMIUM_BIN = os.environ.get("CHROMIUM_BIN")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

SKIP_EXTENSIONS = frozenset(
    {
        ".bib",
        ".ris",
        ".enw",
        ".nbib",
        ".avi",
        ".mp4",
        ".mov",
        ".wmv",
        ".zip",
        ".tar",
        ".gz",
        ".csv",
        ".xlsx",
        ".xls",
        ".tsv",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".svg",
        ".pptx",
        ".ppt",
    }
)

ERROR_SIGNALS = frozenset(
    {
        "403 forbidden",
        "access denied",
        "access to this resource",
        "javascript is disabled",
        "enable javascript",
        "please enable",
        "unusual traffic",
        "captcha",
        "bot detection",
        "you have been blocked",
        "not authorized",
        "verify that you",
        "doi not found",
        "page not found",
        "404 not found",
    }
)

_HTTPX_TIMEOUT = httpx.Timeout(connect=10, read=30, write=10, pool=5)


@dataclass
class FetchResult:
    url: str
    layer: str = ""  # SKIP | API | L1 | L2 | L3 | FAIL
    success: bool = False
    content_type: str = ""
    pdf_bytes: int = 0  # raw PDF size; 0 when no PDF obtained
    text: str = ""  # extracted text (PDF→text or L3 HTML→text)
    final_url: str = ""
    notes: list[str] = field(default_factory=list)
    elapsed: float = 0.0


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _find_pdf_url(html_bytes: bytes, base_url: str) -> str | None:
    try:
        tree = lxml_html.fromstring(html_bytes)
    except Exception:
        return None

    for meta in tree.xpath('//meta[@name="citation_pdf_url"]'):
        val = meta.get("content", "").strip()
        if val:
            return val

    for link in tree.xpath('//link[@rel="alternate"][@type="application/pdf"]'):
        val = link.get("href", "").strip()
        if val:
            return urljoin(base_url, val)

    for a in tree.xpath('//a[contains(@href, ".pdf")]'):
        href = a.get("href", "").strip()
        if href and not href.startswith("#"):
            return urljoin(base_url, href)

    for a in tree.xpath(
        '//a[contains(translate(@class,"PDF","pdf"),"pdf") or '
        'contains(translate(@id,"PDF","pdf"),"pdf")]'
    ):
        href = a.get("href", "").strip()
        if href and not href.startswith("#"):
            return urljoin(base_url, href)

    return None


def _html_to_text(html_bytes: bytes) -> str:
    try:
        tree = lxml_html.fromstring(html_bytes)
        for tag in tree.xpath("//script|//style|//nav|//header|//footer"):
            p = tag.getparent()
            if p is not None:
                p.remove(tag)
        return " ".join(tree.text_content().split())
    except Exception:
        return ""


def _is_error(text: str) -> bool:
    lower = text.lower()
    return any(sig in lower for sig in ERROR_SIGNALS)


def _is_pdf(content: bytes, content_type: str) -> bool:
    return content[:4] == b"%PDF" or "application/pdf" in content_type


def _ensure_libs() -> None:
    """Preload Nix-store libraries that cv2 (docling table structure) needs.

    glibc caches the dlopen search path at process start, so os.environ changes
    made later have no effect. ctypes.CDLL forces the library into the loader's
    link map before the import chain reaches cv2.

    Libraries needed by cv2.abi3.so (as reported by ldd):
      libz, libxcb, libGL, libglib-2.0, libgthread-2.0
    """
    import ctypes
    import glob

    _PRELOAD: list[tuple[str, str]] = [
        ("/nix/store/*-zlib-*/lib/libz.so.1", "libz.so.1"),
        ("/nix/store/*-libxcb-*/lib/libxcb.so.1", "libxcb.so.1"),
        ("/nix/store/*-libglvnd-*/lib/libGL.so.1", "libGL.so.1"),
        ("/nix/store/*-glib-*/lib/libglib-2.0.so.0", "libglib-2.0.so.0"),
        ("/nix/store/*-glib-*/lib/libgthread-2.0.so.0", "libgthread-2.0.so.0"),
    ]

    for pattern, soname in _PRELOAD:
        try:
            ctypes.CDLL(soname)  # already in link map — no-op
        except OSError:
            matches = sorted(glob.glob(pattern))
            if matches:
                try:
                    ctypes.CDLL(matches[0])
                except OSError:
                    pass


def _make_docling_converter():
    """Build a DocumentConverter with OCR disabled (table structure enabled)."""
    _ensure_libs()

    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True
    return DocumentConverter(
        format_options={
            "pdf": PdfFormatOption(
                pipeline_options=opts, backend=PyPdfiumDocumentBackend
            )
        }
    )


def _pdf_to_markdown(data: bytes) -> str:
    """Parse PDF bytes → Markdown via docling. Returns empty string on failure."""
    try:
        import io
        from docling_core.types.io import DocumentStream

        converter = _make_docling_converter()
        stream = DocumentStream(name="paper.pdf", stream=io.BytesIO(data))
        result = converter.convert(stream)
        return result.document.export_to_markdown()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Zenodo API
# ---------------------------------------------------------------------------


async def _zenodo_api(url: str) -> FetchResult | None:
    """Returns FetchResult if URL is a Zenodo record, None otherwise."""
    m = re.search(r"10\.5281/zenodo\.(\d+)", url) or re.search(
        r"zenodo\.org/(?:records?|doi)/(\d+)", url
    )
    if not m:
        return None

    record_id = m.group(1)
    result = FetchResult(url=url)
    result.notes.append(f"zenodo record={record_id}")
    t0 = time.monotonic()

    try:
        async with httpx.AsyncClient(
            headers=HEADERS,
            follow_redirects=True,
            timeout=_HTTPX_TIMEOUT,
            verify=False,
        ) as client:
            api_resp = await client.get(f"https://zenodo.org/api/records/{record_id}")
            result.notes.append(f"status={api_resp.status_code}")

            if api_resp.status_code != 200:
                result.layer = "FAIL"
                result.elapsed = time.monotonic() - t0
                return result

            files = api_resp.json().get("files", [])
            pdf_files = [f for f in files if f.get("key", "").lower().endswith(".pdf")]
            if not pdf_files:
                result.notes.append("no PDF file in record")
                result.layer = "FAIL"
                result.elapsed = time.monotonic() - t0
                return result

            pdf_url = pdf_files[0].get("links", {}).get("self") or pdf_files[0].get(
                "links", {}
            ).get("download")
            if not pdf_url:
                result.notes.append("no download link")
                result.layer = "FAIL"
                result.elapsed = time.monotonic() - t0
                return result

            resp = await client.get(pdf_url)
            ct = resp.headers.get("content-type", "")
            if _is_pdf(resp.content, ct):
                result.layer = "API"
                result.success = True
                result.pdf_bytes = len(resp.content)
                result.text = _pdf_to_markdown(resp.content)
            else:
                result.notes.append(f"download not PDF: {ct[:30]}")
                result.layer = "FAIL"

    except Exception as e:
        result.notes.append(f"error: {e}")
        result.layer = "FAIL"

    result.elapsed = time.monotonic() - t0
    return result


# ---------------------------------------------------------------------------
# Layer 1: httpx
# ---------------------------------------------------------------------------


async def _layer1(url: str) -> FetchResult:
    result = FetchResult(url=url)
    t0 = time.monotonic()

    try:
        async with httpx.AsyncClient(
            headers=HEADERS,
            follow_redirects=True,
            timeout=_HTTPX_TIMEOUT,
            verify=False,
        ) as client:
            resp = await client.get(url)
            ct = resp.headers.get("content-type", "")
            result.content_type = ct
            result.final_url = str(resp.url)

            if _is_pdf(resp.content, ct):
                result.layer = "L1"
                result.success = True
                result.pdf_bytes = len(resp.content)
                result.text = _pdf_to_markdown(resp.content)
                result.notes.append("direct PDF")

            elif "text/html" in ct:
                pdf_url = _find_pdf_url(resp.content, str(resp.url))
                if pdf_url:
                    result.notes.append(f"HTML→pdf: {pdf_url[:70]}")
                    resp2 = await client.get(pdf_url)
                    ct2 = resp2.headers.get("content-type", "")
                    if _is_pdf(resp2.content, ct2):
                        result.layer = "L1"
                        result.success = True
                        result.pdf_bytes = len(resp2.content)
                        result.text = _pdf_to_markdown(resp2.content)
                    else:
                        result.notes.append(f"pdf link → {ct2[:30]}")
                else:
                    result.notes.append("HTML: no PDF link")
            else:
                result.notes.append(f"unexpected ct: {ct[:40]}")

    except Exception as e:
        result.notes.append(f"error: {e}")

    result.elapsed = time.monotonic() - t0
    return result


# ---------------------------------------------------------------------------
# Layer 2: Playwright + stealth
# ---------------------------------------------------------------------------


async def _layer2(prev: FetchResult) -> FetchResult:
    result = FetchResult(url=prev.url, notes=list(prev.notes))
    t0 = time.monotonic()
    pdf_data: list[bytes] = []

    try:
        from playwright.async_api import async_playwright
        from playwright_stealth import Stealth

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                executable_path=CHROMIUM_BIN or None,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            context = await browser.new_context(
                user_agent=HEADERS["User-Agent"],
                viewport={"width": 1280, "height": 800},
                locale="en-US",
            )
            page = await context.new_page()
            await Stealth().apply_stealth_async(page)  # type: ignore[attr-defined]

            async def intercept_pdf(response) -> None:
                if "application/pdf" in response.headers.get("content-type", ""):
                    try:
                        pdf_data.append(await response.body())
                    except Exception:
                        pass

            page.on("response", intercept_pdf)

            try:
                resp = await page.goto(prev.url, timeout=25000, wait_until="load")
                result.final_url = page.url
                ct = resp.headers.get("content-type", "") if resp else ""
                result.content_type = ct

                if pdf_data:
                    result.layer = "L2"
                    result.success = True
                    result.pdf_bytes = len(pdf_data[0])
                    result.text = _pdf_to_markdown(pdf_data[0])
                    result.notes.append("PDF intercepted")
                elif "application/pdf" in ct:
                    body = await resp.body()
                    result.layer = "L2"
                    result.success = True
                    result.pdf_bytes = len(body)
                    result.text = _pdf_to_markdown(body)
                    result.notes.append("direct PDF")
                else:
                    pdf_link: str | None = await page.evaluate("""() => {
                        const m = document.querySelector('meta[name="citation_pdf_url"]');
                        if (m) return m.content;
                        const al = document.querySelector('link[rel="alternate"][type="application/pdf"]');
                        if (al) return al.href;
                        const a = [...document.querySelectorAll('a')].find(a =>
                            a.href && (a.href.endsWith('.pdf') || a.href.includes('/pdf/'))
                        );
                        return a ? a.href : null;
                    }""")
                    if pdf_link:
                        result.notes.append(f"DOM→pdf: {pdf_link[:70]}")
                        await page.goto(pdf_link, timeout=20000)
                        if pdf_data:
                            result.layer = "L2"
                            result.success = True
                            result.pdf_bytes = len(pdf_data[0])
                            result.text = _pdf_to_markdown(pdf_data[0])
                        else:
                            result.notes.append("pdf link did not yield PDF")
                    else:
                        result.notes.append("no PDF found in rendered page")

            except Exception as e:
                result.notes.append(f"navigation error: {e}")
            finally:
                await browser.close()

    except Exception as e:
        result.notes.append(f"L2 error: {e}")

    result.elapsed = time.monotonic() - t0
    return result


# ---------------------------------------------------------------------------
# Layer 3: HTML text fallback
# ---------------------------------------------------------------------------


async def _layer3(prev: FetchResult) -> FetchResult:
    result = FetchResult(url=prev.url, notes=list(prev.notes))
    t0 = time.monotonic()

    try:
        async with httpx.AsyncClient(
            headers=HEADERS,
            follow_redirects=True,
            timeout=httpx.Timeout(connect=10, read=15, write=10, pool=5),
            verify=False,
        ) as client:
            resp = await client.get(prev.url)
            ct = resp.headers.get("content-type", "")
            result.content_type = ct
            result.final_url = str(resp.url)
            text = _html_to_text(resp.content)

            if _is_error(text):
                result.layer = "FAIL"
                result.notes.append(f"error page: {text[:80]!r}")
            elif len(text) < 200:
                result.layer = "FAIL"
                result.notes.append(f"too short ({len(text)} chars)")
            else:
                result.layer = "L3"
                result.success = True
                result.text = text
                result.notes.append(f"text: {text[:60]!r}")

    except Exception as e:
        result.layer = "FAIL"
        result.notes.append(f"error: {e}")

    result.elapsed = time.monotonic() - t0
    return result


# ---------------------------------------------------------------------------
# Source-specific fetchers (used by get_full_text fallback chain)
# ---------------------------------------------------------------------------


async def fetch_pmc(pmc_id: str) -> FetchResult:
    """Fetch full text from PubMed Central.

    Tries PDF first (direct download), then HTML full-text page.
    pmc_id: "PMC3018440" or numeric "3018440" — both accepted.
    """
    num = pmc_id.lstrip("PMCpmc").strip()
    # PMC PDF is not always available but is clean when it exists
    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{num}/pdf/"
    r = await _layer1(pdf_url)
    if r.success:
        r.notes.insert(0, f"PMC{num} pdf")
        return r
    # Fallback: HTML full-text page (confirmed working, L3 text)
    html_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{num}/"
    r2 = await _layer1(html_url)
    if r2.success:
        r2.notes.insert(0, f"PMC{num} html")
        return r2
    r3 = await _layer3(FetchResult(url=html_url))
    r3.notes.insert(0, f"PMC{num} html-text")
    return r3


async def fetch_unpaywall(doi: str) -> FetchResult:
    """Resolve best OA URL via Unpaywall API, then run fetch() on it.

    Uses the public Unpaywall API (no key required, email identifies caller).
    """
    result = FetchResult(url=f"https://doi.org/{doi}")
    t0 = time.monotonic()

    try:
        async with httpx.AsyncClient(
            headers=HEADERS,
            follow_redirects=True,
            timeout=_HTTPX_TIMEOUT,
            verify=False,
        ) as client:
            resp = await client.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": "quarry@example.com"},
            )
            if resp.status_code != 200:
                result.layer = "FAIL"
                result.notes.append(f"unpaywall status={resp.status_code}")
                result.elapsed = time.monotonic() - t0
                return result

            data = resp.json()
            loc = data.get("best_oa_location") or {}
            pdf_url: str | None = loc.get("url_for_pdf")
            landing_url: str | None = loc.get("url_for_landing_page") or loc.get("url")

            oa_url = pdf_url or landing_url
            if not oa_url:
                result.layer = "FAIL"
                result.notes.append("unpaywall: no OA location found")
                result.elapsed = time.monotonic() - t0
                return result

            result.notes.append(f"unpaywall -> {oa_url[:70]}")

    except Exception as e:
        result.layer = "FAIL"
        result.notes.append(f"unpaywall error: {e}")
        result.elapsed = time.monotonic() - t0
        return result

    result.elapsed = time.monotonic() - t0
    # Delegate to full pipeline
    r = await fetch(oa_url)
    r.notes = result.notes + r.notes
    return r


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch(url: str) -> FetchResult:
    """Fetch full text / PDF from a URL using the waterfall pipeline."""
    # SKIP
    path = urlparse(url).path.lower()
    if any(path.endswith(ext) for ext in SKIP_EXTENSIONS):
        r = FetchResult(url=url, layer="SKIP")
        r.notes.append(f"non-document extension ({path.rsplit('.', 1)[-1]})")
        return r

    # API — Zenodo
    r_api = await _zenodo_api(url)
    if r_api is not None:
        if r_api.success:
            return r_api
        # Recognized zenodo record but failed — fall through to L1

    # L1 — httpx
    r1 = await _layer1(url)
    if r1.success:
        return r1

    # L2 — Playwright stealth
    r2 = await _layer2(r1)
    if r2.success:
        return r2

    # L3 — HTML text fallback
    r3 = await _layer3(r1)
    return r3
