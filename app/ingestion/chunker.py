from dataclasses import dataclass, field
from bs4 import BeautifulSoup, NavigableString
from app.ingestion.cleaner import fix_encoding

HEADER_TAGS = ["b", "strong", "h1", "h2", "h3", "h4"]
MAX_HEADER_WORDS = 6


@dataclass
class Chunk:
    text: str
    section_label: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _is_header(text: str) -> bool:
    text = text.strip()
    if not text or len(text) > 60 or len(text.split()) > MAX_HEADER_WORDS:
        return False
    if not text.rstrip(":").strip():
        return False  # e.g. a bold ":" used as a visual separator, not a real label
    return text.endswith(":") or text.istitle() or text.isupper()


def _extract_sections(html: str) -> list[tuple[str, str]]:
   
    soup = BeautifulSoup(fix_encoding(html), "lxml")
    sections, label, buffer = [], "Overview", []

    def flush():
        text = " ".join(buffer).strip()
        if text:
            sections.append((label, text))

    for node in soup.descendants:
        if isinstance(node, NavigableString):
            if node.parent and node.parent.name in HEADER_TAGS:
                continue  # this string IS the header's own label text, not content
            text = node.strip()
            if text:
                buffer.append(text)
        elif node.name in HEADER_TAGS and _is_header(node.get_text()):
            flush()
            label, buffer = node.get_text().strip().rstrip(":"), []
    flush()
    return sections


def _sliding_window(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [text.strip()]
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step) if words[i:i + size]]


def chunk_job_description(raw_html: str, chunk_size_tokens: int = 300, overlap_tokens: int = 50) -> list[Chunk]:
    if not isinstance(raw_html, str) or not raw_html.strip():
        return []

    sections = _extract_sections(raw_html)
    if len(sections) < 2:
        text = BeautifulSoup(fix_encoding(raw_html), "lxml").get_text(" ", strip=True)
        sections = [("general", text)]

    chunks, idx = [], 0
    for label, text in sections:
        for piece in _sliding_window(text, chunk_size_tokens, overlap_tokens):
            chunks.append(Chunk(text=piece, section_label=label, chunk_index=idx))
            idx += 1
    return chunks