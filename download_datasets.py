import os
import requests
import pyarrow.parquet as pq
from tqdm import tqdm


WIKITEXT_URLS = [
    "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet",
    "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-raw-v1/train-00001-of-00002.parquet",
]


def download_parquet(url: str, dest: str, retries=20):
    if os.path.exists(dest):
        print(f"  {dest} exists, skipping")
        return
    for attempt in range(retries):
        try:
            print(f"  Downloading {url}")
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
            print(f"  Saved {dest} ({len(resp.content) / 1e6:.1f} MB)")
            return
        except requests.ConnectionError as e:
            wait = (2 ** (attempt + 2)) + __import__("random").uniform(0, 5)
            print(f"  Connection error (attempt {attempt+1}): waiting {wait:.0f}s...")
            __import__("time").sleep(wait)
    raise RuntimeError(f"Failed to download {url}")


def extract_wikitext(corpus_dir: str, parquet_dir: str):
    out = os.path.join(corpus_dir, "_wikitext103.md")
    count = 0
    for url in WIKITEXT_URLS:
        fname = url.rsplit("/", 1)[-1]
        local = os.path.join(parquet_dir, fname)
        table = pq.read_table(local)
        n = len(table)
        texts = []
        for i in tqdm(range(n), desc=f"  {fname[:48]}", unit="articles"):
            text = table["text"][i].as_py().strip()
            if text:
                texts.append(text)
        with open(out, "a", encoding="utf-8", newline="\n") as f:
            f.write("\n\n".join(texts) + "\n\n")
        count += len(texts)
    print(f"Wikitext-103: {count} articles -> {out}")


def main():
    corpus_dir = "data/corpus"
    parquet_dir = "data/parquet"
    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(parquet_dir, exist_ok=True)

    print("Downloading Wikitext-103 parquet files...")
    for url in WIKITEXT_URLS:
        fname = url.rsplit("/", 1)[-1]
        download_parquet(url, os.path.join(parquet_dir, fname))

    print("\nExtracting Wikitext-103 to corpus...")
    extract_wikitext(corpus_dir, parquet_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
