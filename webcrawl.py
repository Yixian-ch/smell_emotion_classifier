from dataclasses import dataclass, field, asdict
import requests
import json
from pathlib import Path
import argparse
from typing import List
from tqdm import tqdm
import random
import time


@dataclass
class Article:
    id: str                      
    title: str                  
    doc_url: str              
    excerpt: str            
    emotions: List[str] = field(default_factory=list)

@dataclass
class Corpus:
    articles: List[Article] = field(default_factory=list)


def get_json_data(api_url):
    response = requests.get(api_url)
    data = response.json()
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


def extract_data(json_path: Path) -> Corpus:
    corpus = Corpus()
    with json_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    for doc_id, result in enumerate(raw_data["results"], start=1):
        source_dict = result.get("source", {})
        title = source_dict.get("label", None)
        doc_url = source_dict.get("url", None)
        excerpt = result.get("text", None)
        emotions_raw = result.get("emotion", [])
        emotions = [emotion.get("label", "") for emotion in emotions_raw if "label" in emotion]

        article = Article(
            id=str(doc_id),
            title=title,
            doc_url=doc_url,
            excerpt=excerpt,
            emotions=emotions
        )

        corpus.articles.append(article)

    return corpus

def save_json(corpus:Corpus, output_file:Path) -> None:
    with output_file.open("w",encoding="utf-8") as fp:
        json.dump(asdict(corpus), fp, ensure_ascii=False, indent=6, sort_keys=True)

def load_json(input_file: Path) -> Corpus:
    with input_file.open("r", encoding="utf-8") as fp:
        data = json.load(fp)  

    corpus = Corpus()
    for article_dict in data.get("articles", []):
        article_obj = Article(
            id=article_dict["id"],
            title=article_dict["title"],
            doc_url=article_dict.get("doc_url", ""),
            excerpt=article_dict.get("excerpt", ""),
            emotions=article_dict.get("emotions", []),
            
        )
        corpus.articles.append(article_obj)

    return corpus

def process_url(url: str, raw_output_path: Path, data_output_path: Path) -> None:
    raw_data = get_json_data(url)
    with raw_output_path.open("w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

    corpus = extract_data(raw_output_path)
    save_json(corpus, data_output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", help="Base API URL with {page} placeholder.")
    parser.add_argument("output_directory", help="Folder to save raw and data outputs")
    args = parser.parse_args()

    base_url = args.base_url
    output_dir = Path(args.output_directory)

    raw_data = output_dir / "raw"
    data = output_dir / "data"
    raw_data.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    for page in tqdm(range(1, 301), desc="Processing pages"):
        full_url = base_url.format(page=page)
        raw_output_path = Path(raw_data) / f"raw_page_{page:03}.json"
        data_output_path = Path(data) / f"data_page_{page:03}.json"

        process_url(full_url, raw_output_path, data_output_path)
        # sleep_time = round(random.uniform(0.5, 1.5), 2)
        print(f"Page {page} saved.")
        # time.sleep(sleep_time)

    print("\nDone! Files saved in:")
    print(f"Raw API data:{raw_data}")
    print(f"Structured data:{data}")

if __name__=="__main__":
    main()
