import os
import json
import time
import random
import hashlib
import requests
from tqdm import tqdm

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "TinyGPT/1.0 (educational; contact@example.com)"}
OUT_DIR = "data/corpus"
PROGRESS_FILE = "data/scrape_progress.json"
TARGET_FILES = 10000
BASE_SLEEP = 1.0


def api_call(params, retries=5):
    for attempt in range(retries):
        try:
            resp = requests.get(API, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                wait = (2 ** attempt) + random.uniform(0, 2)
                tqdm.write(f"429: waiting {wait:.0f}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            wait = (3 ** attempt) + random.uniform(1, 5)
            tqdm.write(f"Connection error (attempt {attempt+1}): waiting {wait:.0f}s...")
            time.sleep(wait)
            continue
    raise RuntimeError("Max retries exceeded")


def get_random_titles(count=50):
    data = api_call({
        "action": "query",
        "format": "json",
        "list": "random",
        "rnlimit": min(count, 50),
        "rnnamespace": 0,
    })
    return [p["title"] for p in data["query"]["random"]]


def get_batch_extracts(titles):
    titles = titles[:50]
    data = api_call({
        "action": "query",
        "format": "json",
        "titles": "|".join(titles),
        "prop": "extracts",
        "explaintext": True,
        "exlimit": 50,
        "exintro": False,
    })
    results = {}
    for pid, page in data["query"]["pages"].items():
        if "extract" in page and len(page["extract"]) > 500:
            results[page["title"]] = page["extract"]
    return results


def slug(title):
    s = title.strip().replace(" ", "_")
    s = "".join(c for c in s if c.isalnum() or c in "_-")
    if not s:
        s = "article"
    return s[:120]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    seen = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            seen = set(json.load(f))

    existing = {f.replace(".md", "") for f in os.listdir(OUT_DIR) if f.endswith(".md")}
    print(f"Already have {len(existing)} articles. Target: {TARGET_FILES}")

    pbar = tqdm(total=TARGET_FILES, initial=len(existing))
    fail_streak = 0

    while len(existing) < TARGET_FILES:
        titles = get_random_titles(50)
        fail_streak += 1

        batch = []
        for title in titles:
            key = hashlib.md5(title.encode()).hexdigest()
            if key not in seen and slug(title) not in existing:
                batch.append(title)

        if not batch:
            time.sleep(BASE_SLEEP)
            continue

        extracts = get_batch_extracts(batch)
        for title, text in extracts.items():
            key = hashlib.md5(title.encode()).hexdigest()
            seen.add(key)
            fname = slug(title)
            if fname in existing:
                continue
            md = f"# {title}\n\n{text}\n"
            path = os.path.join(OUT_DIR, f"{fname}.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(md)
            existing.add(fname)
            pbar.update(1)
            fail_streak = 0
            if len(existing) >= TARGET_FILES:
                break

        for title in batch:
            key = hashlib.md5(title.encode()).hexdigest()
            seen.add(key)

        if fail_streak > 5:
            tqdm.write("Too many failures, sleeping longer...")
            time.sleep(30)
            fail_streak = 0

        time.sleep(BASE_SLEEP + random.uniform(0, 2))

        with open(PROGRESS_FILE, "w") as f:
            json.dump(list(seen), f)

    pbar.close()
    total_size = sum(
        os.path.getsize(os.path.join(OUT_DIR, f))
        for f in os.listdir(OUT_DIR) if f.endswith(".md")
    )
    print(f"Done. {len(existing)} articles, {total_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
