from pathlib import Path
from webcrawl import load_json, save_json, Corpus
import argparse
import pandas as pd

def get_args():
    """
    TASK 1: Reworks the existing JSON files and keeps only the articles that have the target emotion as their emotion.
    CLI: python filter_data-emotion.py -ip data/Odeuropa_output-surprise/data -op data/filtered_data/surprise -e surprise -t filter_corpus

    TASK 2: Iterates through the reworked JSON files, taking the excerpts in each JSON file, concatenating them, and assigning it as the text for that page.
    Finally, we'll have a CSV file in which each row is a text reflecting all the texts in the JSON file.
    CLI: python filter_data-emotion.py -ip data/filtered_data/surprise -op data/filtered_data/surprise -e surprise -t json_to_csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", "--input_path", type=str, required=True)
    parser.add_argument("-op", "--output_path", type=str, required=True)
    parser.add_argument("-e", "--emotion", type=str, required=True)
    parser.add_argument("-t", "--task", type=str, choices=["filter_corpus", "json_to_csv"], required=True)
    args = parser.parse_args()
    return args

def filter_corpus(emotion: str, input_folder: Path, output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    paths = input_folder.rglob("*.json")
    global_article_counter = 0
    global_emotion_counter = 0

    for file_path in paths:
        file_name = file_path.stem
        corpus = load_json(file_path)
        articles = corpus.articles

        filtered_corpus = Corpus()

        for article in articles:
            global_article_counter += 1
            if article.emotions == [emotion]:
                global_emotion_counter += 1
                filtered_corpus.articles.append(article)

        output_path = output_folder / f"{file_name}.json"
        save_json(filtered_corpus, output_path)

    print(f"Total articles processed: {global_article_counter}")
    print(f"Articles with emotion '{emotion}': {global_emotion_counter}")

def json_to_csv(emotion: str, filtered_data_folder: Path, output_folder: Path) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    filtered_data = []
    paths = filtered_data_folder.rglob("*.json")

    for file_path in paths:
        file_name = file_path.stem
        corpus = load_json(file_path)
        articles = corpus.articles

        concatenated_excerpts = ""
        for article in articles:
            excerpt = article.excerpt

            if not excerpt:
                continue

            if isinstance(excerpt, list):
                concatenated_excerpts += " " + " ".join(excerpt)
            else:
                concatenated_excerpts += " " + excerpt

        if concatenated_excerpts:
            filtered_data.append({
                "id": file_name,
                "emotion": emotion,
                "text": concatenated_excerpts,
            })

    df = pd.DataFrame(filtered_data)
    df.index += 1
    output_path = output_folder / f"{emotion}.csv"
    print(f"Total rows in {emotion}.csv: {df.shape[0]}")
    df.to_csv(output_path, index=True, header=True)

def main():
    args = get_args()
    input_folder_path = Path(args.input_path)
    output_folder_path = Path(args.output_path)
    emotion = args.emotion
    task = args.task

    if task == "filter_corpus":
        filter_corpus(emotion, input_folder_path, output_folder_path)
    elif task == "json_to_csv":
        json_to_csv(emotion, input_folder_path, output_folder_path)

if __name__ == "__main__":
    main()
