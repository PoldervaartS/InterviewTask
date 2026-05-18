import re

import httpx
from html.parser import HTMLParser

_SKIP_TAGS = frozenset({
    "script", "style", "nav", "footer", "header",
    "noscript", "aside", "meta", "link", "iframe",
})
_MAX_CHARS = 40_000
_WIKIPEDIA_HOST_RE = re.compile(r"https?://[a-z]+\.wikipedia\.org/")
# Contact info required by Wikimedia bot policy: https://meta.wikimedia.org/wiki/User-Agent_policy
_USER_AGENT = "QuizBot/1.0 (srpolde@sandia.gov; educational use)"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        return "\n".join(self._parts)


def _wikipedia_title(url: str) -> str:
    """Extract the article title from a Wikipedia URL (any common form)."""
    # /wiki/Title form
    m = re.search(r"/wiki/([^?#]+)", url)
    if m:
        return m.group(1).replace("_", " ")
    # index.php?title=Title form
    m = re.search(r"[?&]title=([^&#]+)", url)
    if m:
        return m.group(1).replace("_", " ")
    raise ValueError(f"Cannot extract Wikipedia title from URL: {url}")


def _fetch_wikipedia(url: str) -> str:
    """Use the Wikipedia action=query API to retrieve plain-text article content."""
    title = _wikipedia_title(url)
    api_url = re.sub(r"(https?://[a-z]+\.wikipedia\.org/).*", r"\1w/api.php", url)
    headers = {"User-Agent": _USER_AGENT}
    with httpx.Client(follow_redirects=True, timeout=20) as client:
        response = client.get(
            api_url,
            params={
                "action": "query",
                "titles": title,
                "prop": "extracts",
                "explaintext": True,
                "format": "json",
            },
            headers=headers,
        )
    response.raise_for_status()
    pages = response.json()["query"]["pages"]
    page = next(iter(pages.values()))
    return page.get("extract", "")


def fetch_text(url: str) -> str:
    """Fetch a URL and return extracted plain text, truncated to _MAX_CHARS."""
    if _WIKIPEDIA_HOST_RE.match(url):
        return _fetch_wikipedia(url)[:_MAX_CHARS]

    headers = {"User-Agent": _USER_AGENT}
    with httpx.Client(follow_redirects=True, timeout=20) as client:
        response = client.get(url, headers=headers)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        extractor = _TextExtractor()
        extractor.feed(response.text)
        text = extractor.get_text()
    else:
        text = response.text

    return text[:_MAX_CHARS]
