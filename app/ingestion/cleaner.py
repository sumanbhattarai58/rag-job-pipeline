import re
import ftfy
from bs4 import BeautifulSoup

BLOCK_TAGS = ["p", "li", "div", "br", "tr", "h1", "h2", "h3", "h4", "ul", "ol", "table"]


def fix_encoding(text: str) -> str:
    """Repair mojibake (e.g. 'â€™' -> ''', 'Â¿' -> stray char removed)."""
    if not isinstance(text, str):
        return text
    return ftfy.fix_text(text)


def html_to_text(html: str) -> str:
    """Convert an HTML job description into clean, readable plain text."""
    if not isinstance(html, str) or not html.strip():
        return ""

    soup = BeautifulSoup(html, "lxml")

    for tag_name in BLOCK_TAGS:
        for tag in soup.find_all(tag_name):
            tag.append("\n")

    text = soup.get_text()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def clean_job_description(raw_html: str) -> str:
    """Full cleaning pipeline: fix encoding, then strip HTML to plain text."""
    fixed = fix_encoding(raw_html)
    return html_to_text(fixed)


def clean_field(value) -> str:
    """Clean a short text field (title, company, location, etc.) — encoding fix only."""
    if value is None:
        return ""
    try:
        import pandas as pd
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return fix_encoding(str(value)).strip()