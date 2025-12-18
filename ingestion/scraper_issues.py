import requests
import os
import json
import time
import hashlib
from dotenv import load_dotenv

OWNER = "pytorch"
REPO = "pytorch"
LABELS = ["question", "docs", "bug"]
OUTPUT_DIR = "data/raw/issues"

os.makedirs(OUTPUT_DIR, exist_ok=True)

load_dotenv()
HEADERS = {
    "Authorization": f"token {os.environ['GITHUB_TOKEN']}",
    "Accept": "application/vnd.github+json"
}


def issue_to_id(issue_number: int) -> str:
    return hashlib.md5(f"issue_{issue_number}".encode()).hexdigest()


def get_issues(label, max_pages=5):
    issues = []
    for page in range(1, max_pages + 1):
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues"
        params = {
            "state": "closed",
            "labels": label,
            "per_page": 30,
            "page": page
        }
        r = requests.get(url, headers=HEADERS, params=params)
        if r.status_code != 200:
            break
        batch = r.json()
        if not batch:
            break
        issues.extend(batch)
        time.sleep(1)
    return issues


def get_comments(comments_url):
    r = requests.get(comments_url, headers=HEADERS)
    if r.status_code != 200:
        return []
    return r.json()


def extract_best_answer(comments):
    if not comments:
        return None, None

    # Take the LAST non-bot comment
    for c in reversed(comments):
        body = c["body"].strip()
        if len(body) > 50:
            return body, c["user"]["login"]

    # fallback: longest comment
    best = max(comments, key=lambda c: len(c["body"].strip()))
    return best["body"], best["user"]["login"]



if __name__ == "__main__":
    for label in LABELS:
        issues = get_issues(label)
        for issue in issues:
          if "pull_request" in issue and issue["comments"] == 0:
              continue

          comments = get_comments(issue["comments_url"])
          answer, author = extract_best_answer(comments)

          if not answer:
              continue

          doc = {
              "doc_id": issue_to_id(issue["number"]),
              "source": "github_issue",
              "title": issue["title"],
              "text": f"Question:\n{issue['body']}\n\nAnswer:\n{answer}",
              "url": issue["html_url"],
              "metadata": {
                  "section": None,
                  "issue_number": issue["number"],
                  "labels": [l["name"] for l in issue["labels"]],
                  "answer_author": author
              }
          }

          out_path = os.path.join(OUTPUT_DIR, f"{doc['doc_id']}.json")
          with open(out_path, "w") as f:
              json.dump(doc, f, indent=2)

