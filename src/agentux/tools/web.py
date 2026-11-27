"""Live technical web search and page content extraction tools.

Enables the agent to search online documentation, code references,
and error solutions without third-party python dependencies.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from ..errors import ToolError
from .base import BaseTool

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Agentux/0.2.0"

#: Words that carry no subject, so they must not drive a Wikipedia lookup.
_WIKI_STOPWORDS = frozenset(
    {
        "what", "who", "whom", "whose", "when", "where", "why", "how", "which",
        "is", "are", "was", "were", "do", "does", "did", "can", "could",
        "should", "would", "will", "the", "and", "for", "with", "that",
        "this", "from", "tell", "explain", "please", "about", "your", "you",
    }
)


class WebSearchTool(BaseTool):
    """Search technical documentation and code solutions on the live web."""

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search technical documentation, error messages, and API references "
            "on the live web. Uses DuckDuckGo HTML search by default, or Tavily/Brave "
            "if API keys are configured."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Technical search query (e.g. 'pydantic v2 field_validator example').",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return (default: 5).",
                },
            },
            "required": ["query"],
        }

    def search(self, query: str, max_results: int = 5) -> str:
        """Run a search synchronously.

        Exists so callers outside the async tool path -- Agentux Solo grounds a
        question before the model answers it -- can use it without an event
        loop. ``execute`` is the tool-facing wrapper.
        """
        query = query.strip()
        if not query:
            raise ToolError("Query cannot be empty.")

        # Check for Tavily API key
        tavily_key = os.environ.get("TAVILY_API_KEY")
        if tavily_key:
            return self._search_tavily(query, tavily_key, max_results)

        # Check for Brave API key
        brave_key = os.environ.get("BRAVE_API_KEY")
        if brave_key:
            return self._search_brave(query, brave_key, max_results)

        # Default: Zero-dependency DuckDuckGo HTML scraper
        return self._search_duckduckgo(query, max_results)

    async def execute(self, query: str, max_results: int = 5, **kwargs: Any) -> str:
        if "num_results" in kwargs:
            max_results = int(kwargs["num_results"])
        return await asyncio.to_thread(self.search, query, max_results)

    def _fetch(self, url: str) -> str:
        """GET a URL and return the decoded body."""
        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/html"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def search_wikipedia(self, query: str, max_chars: int = 1200) -> str:
        """Look a topic up on Wikipedia and return its lead sentence.

        Keyless and reliable, which makes it the best grounding source for a
        small model: it returns a sentence containing the fact rather than a
        headline the model can misread.

        Each candidate phrase is searched on its own and the best-matching
        article wins. Opensearch matches whole phrases, so a question finds
        nothing while its content words find the article; a single word often
        lands on a disambiguation page ("Mojo" is one), so the phrase is tried
        first. A title that merely contains the word is not enough either --
        the extract has to mention it, and a short title beats a long one when
        the mentions tie.
        """
        words = self._subject_words(query)
        if not words:
            return ""
        terms = set(words)

        candidates: List[str] = []
        phrase = " ".join(words)
        if phrase:
            candidates.append(phrase)
        candidates.extend(word for word in words if word != phrase)

        best_score = None
        best_title = ""
        best_extract = ""
        try:
            for candidate in candidates[:4]:
                for title in self._wikipedia_titles(candidate)[:2]:
                    extract = self._wikipedia_summary(title)
                    if not extract:
                        continue
                    haystack = (title + " " + extract).lower()
                    hits = sum(1 for term in terms if term in haystack)
                    score = (hits, -len(title.split()))
                    if best_score is None or score > best_score:
                        best_score = score
                        best_title = title
                        best_extract = extract
            if not best_extract:
                return ""
            # The lead paragraph may not answer the question -- "who invented
            # the transistor" is not answered by a definition. Pick the
            # paragraph that mentions the subject words most.
            paragraph = self._wikipedia_best_paragraph(best_title, list(terms))
            return (paragraph or best_extract)[:max_chars]
        except Exception:
            return ""

    def _wikipedia_best_paragraph(self, title: str, terms: List[str]) -> str:
        """The paragraph of an article that mentions the subject words most."""
        text = self._wikipedia_full_extract(title)
        if not text:
            return ""
        best = ""
        best_hits = 0
        for paragraph in text.split("\n"):
            clean = " ".join(paragraph.split())
            if len(clean) < 40:
                continue
            lowered = clean.lower()
            hits = sum(1 for term in terms if term in lowered)
            if hits > best_hits:
                best, best_hits = clean, hits
        return best

    def _wikipedia_full_extract(self, title: str) -> str:
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "format": "json",
                "redirects": 1,
                "titles": title,
            }
        )
        data = json.loads(self._fetch(url))
        for page in data.get("query", {}).get("pages", {}).values():
            extract = page.get("extract")
            if extract:
                return str(extract)
        return ""

    @staticmethod
    def _subject_words(query: str) -> List[str]:
        """The words in a question that could name its subject, in order."""
        return [
            word
            for word in re.findall(r"[a-z0-9]+", str(query).lower())
            if len(word) >= 4 and word not in _WIKI_STOPWORDS
        ]

    def _wikipedia_titles(self, query: str) -> List[str]:
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {
                "action": "opensearch",
                "search": query,
                "limit": 5,
                "format": "json",
                "profile": "fuzzy",
            }
        )
        data = json.loads(self._fetch(url))
        if len(data) > 1 and isinstance(data[1], list):
            return [str(title) for title in data[1]]
        return []

    def _wikipedia_summary(self, title: str) -> str:
        path = urllib.parse.quote(title.replace(" ", "_"), safe="()_")
        data = json.loads(
            self._fetch(f"https://en.wikipedia.org/api/rest_v1/page/summary/{path}")
        )
        extract = data.get("extract")
        if not extract or data.get("type") == "disambiguation":
            return ""
        return f"{data.get('title', title)}: {extract}"

    def _search_duckduckgo(self, query: str, max_results: int) -> str:
        encoded_query = urllib.parse.urlencode({"q": query, "b": ""})
        url = f"https://html.duckduckgo.com/html/?{encoded_query}"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            return f"Web search failed: {exc}"

        # Parse results with regex
        results: List[Dict[str, str]] = []
        pattern = re.compile(
            r'<a[^>]+class="result__snippet"[^>]*href="(?P<link>[^"]+)"[^>]*>(?P<snippet>.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )
        title_pattern = re.compile(
            r'<a[^>]+class="result__url"[^>]*>(?P<title>.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )

        # Alternative generic extraction for DuckDuckGo HTML layout
        block_pattern = re.compile(
            r'<div[^>]+class="[^"]*result__body[^"]*"[^>]*>(.*?)</div>\s*</div>',
            re.DOTALL | re.IGNORECASE,
        )

        for block in block_pattern.findall(content):
            if len(results) >= max_results:
                break

            title_m = re.search(r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.DOTALL)
            snippet_m = re.search(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', block, re.DOTALL)

            if title_m:
                raw_url = title_m.group(1)
                # Unpack DuckDuckGo redirect uddg= if present
                if "uddg=" in raw_url:
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
                    raw_url = parsed.get("uddg", [raw_url])[0]

                title = re.sub(r"<[^>]+>", "", title_m.group(2)).strip()
                title = html.unescape(title)

                snippet = ""
                if snippet_m:
                    snippet = re.sub(r"<[^>]+>", "", snippet_m.group(1)).strip()
                    snippet = html.unescape(snippet)

                results.append({
                    "title": title or "Untitled",
                    "url": raw_url,
                    "snippet": snippet,
                })

        if not results:
            return f"No results found for query: {query!r}"

        lines = [f"Web Search Results for '{query}' ({len(results)} matches):\n"]
        for idx, item in enumerate(results, 1):
            lines.append(f"{idx}. **{item['title']}**")
            lines.append(f"   URL: {item['url']}")
            if item["snippet"]:
                lines.append(f"   {item['snippet']}")
            lines.append("")

        return "\n".join(lines).strip()

    def _search_tavily(self, query: str, api_key: str, max_results: int) -> str:
        req_data = json.dumps({"api_key": api_key, "query": query, "max_results": max_results}).encode("utf-8")
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=req_data,
            headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return f"Tavily search error: {exc}"

        results = data.get("results", [])
        if not results:
            return f"No results found for query: {query!r}"

        lines = [f"Tavily Search Results for '{query}':\n"]
        for idx, r in enumerate(results[:max_results], 1):
            lines.append(f"{idx}. **{r.get('title', 'Untitled')}**")
            lines.append(f"   URL: {r.get('url', '')}")
            lines.append(f"   {r.get('content', '')}\n")
        return "\n".join(lines).strip()

    def _search_brave(self, query: str, api_key: str, max_results: int) -> str:
        encoded = urllib.parse.urlencode({"q": query, "count": max_results})
        url = f"https://api.search.brave.com/res/v1/web/search?{encoded}"
        req = urllib.request.Request(
            url,
            headers={"X-Subscription-Token": api_key, "Accept": "application/json", "User-Agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return f"Brave search error: {exc}"

        results = data.get("web", {}).get("results", [])
        if not results:
            return f"No results found for query: {query!r}"

        lines = [f"Brave Search Results for '{query}':\n"]
        for idx, r in enumerate(results[:max_results], 1):
            lines.append(f"{idx}. **{r.get('title', 'Untitled')}**")
            lines.append(f"   URL: {r.get('url', '')}")
            lines.append(f"   {r.get('description', '')}\n")
        return "\n".join(lines).strip()


class FetchWebPageTool(BaseTool):
    """Scrape and convert online web pages and documentation to clean readable text."""

    @property
    def name(self) -> str:
        return "fetch_web_page"

    @property
    def description(self) -> str:
        return (
            "Fetch an online URL and extract its text content, converting HTML "
            "into clean readable plain text / markdown. Useful for reading documentation."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full HTTP or HTTPS URL to fetch.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return (default: 8000).",
                },
            },
            "required": ["url"],
        }

    async def execute(self, url: str, max_chars: int = 8000) -> str:
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            raise ToolError(f"URL must start with http:// or https://: {url!r}")

        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                headers = getattr(resp, "headers", None)
                charset = (headers.get_content_charset() if headers and hasattr(headers, "get_content_charset") else None) or "utf-8"
                raw_html = resp.read().decode(charset, errors="replace")
        except Exception as exc:
            return f"Failed to fetch {url}: {exc}"

        text = self._html_to_markdown(raw_html)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[... truncated at {max_chars} characters ...]"

        return f"Content of {url}:\n\n{text}"

    def _html_to_markdown(self, raw_html: str) -> str:
        # Strip script, style, head, nav, footer
        content = re.sub(r"<(script|style|head|nav|footer)[^>]*>.*?</\1>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<!--.*?-->", " ", content, flags=re.DOTALL)

        # Convert links <a href="url">text</a> -> [text](url)
        content = re.sub(
            r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            lambda m: f"[{re.sub(r'<[^>]+>', '', m.group(2)).strip()}]({m.group(1)})",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Convert headings
        for h in range(6, 0, -1):
            content = re.sub(
                rf"<h{h}[^>]*>(.*?)</h{h}>",
                rf"\n\n{'#' * h} \1\n\n",
                content,
                flags=re.DOTALL | re.IGNORECASE,
            )

        # Convert paragraphs, list items, breaks
        content = re.sub(r"<li[^>]*>(.*?)</li>", r"\n• \1", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\n\1\n\n", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<br\s*/?>", "\n", content, flags=re.IGNORECASE)

        # Strip remaining tags
        content = re.sub(r"<[^>]+>", " ", content)

        # Decode HTML entities
        content = html.unescape(content)

        # Collapse whitespace
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in content.split("\n")]
        collapsed = []
        prev_blank = False
        for line in lines:
            if not line:
                if not prev_blank:
                    collapsed.append("")
                    prev_blank = True
            else:
                collapsed.append(line)
                prev_blank = False

        return "\n".join(collapsed).strip()


__all__ = ["WebSearchTool", "FetchWebPageTool"]
