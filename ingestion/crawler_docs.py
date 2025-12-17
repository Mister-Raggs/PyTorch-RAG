import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import json
import hashlib

BASE_URL = "https://pytorch.org/docs/stable/"
OUTPUT_DIR = "data/raw/docs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

visited = set()


def url_to_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def clean_text(soup: BeautifulSoup) -> str:
    # Remove nav, footer, sidebar
    for tag in soup(["nav", "footer", "script", "style"]):
        tag.decompose()

    main = soup.find("main") or soup.body
    text_blocks = []
    if not main:
        return ""
    for elem in main.find_all(["h1", "h2", "h3", "p", "pre", "code"]):
        if elem.name in ["h1", "h2", "h3"]:
            text_blocks.append(f"\n## {elem.get_text(strip=True)}\n")
        elif elem.name == "pre":
            text_blocks.append(f"\n```python\n{elem.get_text()}\n```\n")
        else:
            text_blocks.append(elem.get_text(strip=True))

    return "\n".join(text_blocks)


def crawl(url: str):
    if url in visited:
        return
    visited.add(url)

    print(f"Crawling {url}")
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return

    soup = BeautifulSoup(r.text, "html.parser")

    doc = {
        "doc_id": url_to_id(url),
        "source": "pytorch_docs",
        "title": soup.title.get_text() if soup.title else "",
        "text": clean_text(soup),
        "url": url,
        "metadata": {
            "section": None,
            "issue_number": None,
            "labels": None,
            "answer_author": None
        }
    }

    out_path = os.path.join(OUTPUT_DIR, f"{doc['doc_id']}.json")
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)

    # Follow internal links only
    if not soup:
        return
    for a in soup.find_all("a", href=True):
        next_url = urljoin(url, a["href"])
        parsed = urlparse(next_url)

        if (
            next_url.startswith(BASE_URL)
            and parsed.fragment == ""
            and next_url not in visited
        ):
            crawl(next_url)


if __name__ == "__main__":
    crawl(BASE_URL)
