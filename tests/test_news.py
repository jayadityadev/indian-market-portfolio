from __future__ import annotations

from news import _parse_rss


def test_parse_rss_preserves_source_provenance():
    content = b"""
    <rss><channel><item>
      <title>RBI keeps policy unchanged</title>
      <link>https://example.com/rbi</link>
      <source>Example News</source>
      <pubDate>Sun, 16 Aug 2026 10:00:00 GMT</pubDate>
      <description>Policy decision summary</description>
    </item></channel></rss>
    """
    rows = _parse_rss(content, "https://example.com/feed.xml")
    assert rows == [
        {
            "title": "RBI keeps policy unchanged",
            "url": "https://example.com/rbi",
            "publisher": "Example News",
            "published_at": "Sun, 16 Aug 2026 10:00:00 GMT",
            "snippet": "Policy decision summary",
            "source": "rss",
        }
    ]
