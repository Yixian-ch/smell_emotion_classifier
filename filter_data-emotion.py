from pathlib import Path
from webcrawl import load_json, save_json, Corpus
import argparse
import pandas as pd

def get_args():
    """
    TASK 1: Reworks the existing JSON files and keeps only the articles that have 
    exactly the target emotion (no mixed emotions).
    Example: python filter_data-emotion.py -ip disgust_output/data -op task_1_output/disgust -e disgust -t filter_corpus

    TASK 2: Goes through the filtered JSON files, concatenates the excerpts in each file, 
    and assigns them as the text for that page.
    The result is a CSV file where each row represents the full text from one JSON file.
    Example: python filter_data-emotion.py -ip task_1_output/disgust -op task_2_output -e disgust -t json_to_csv  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", "--input_path", type=str, required=True)
    parser.add_argument("-op", "--output_path", type=str, required=True)
    parser.add_argument("-e", "--emotion", type=str, required=True)
    parser.add_argument("-t", "--task", type=str, choices=["filter_corpus", "json_to_csv"], required=True)
    args = parser.parse_args()
    return args

def filter_corpus(emotion: str, input_folder: Path, output_folder: Path) -> None:
    """
    Filters each JSON file in the input folder (which is the emotion folder) 
    to keep only articles that have exactly the target emotion (no mixed emotions).

    Saves the filtered JSON files to the output folder using the same filenames.
    This prepares the data for task 2: creating the CSV.
    """
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

    """
    Goes through the filtered_data/ folder, where each subfolder is an emotion (like 'fear/'). 
    Each JSON file in that folder represents a page with several articles that all share the same emotion.

    For each file, this function combines all the excerpts from the articles into one string.

    Returns:
        A list of dictionaries. Each dictionary has:
            - 'emotion': the emotion label
            - 'text': the combined excerpts from one file

    Each item in the list represents one JSON file (one page).

    This list is then used to create a CSV file where each row reflects one page of text.
    """

    output_folder.mkdir(parents=True, exist_ok=True)
    filtered_data = []
    paths = filtered_data_folder.rglob("*.json")

    for file_path in paths:
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
                "emotion": emotion,
                "text": concatenated_excerpts,
            })

    df = pd.DataFrame(filtered_data)
    output_path = output_folder / f"{emotion}.csv"
    print(f"Total rows in {emotion}.csv: {df.shape[0]}")
    df.to_csv(output_path, index=False, header=True)

def main():
    args = get_args()
    input_folder_path = Path(args.input_path)
    output_folder_path = Path(args.output_path)
    emotion = args.emotion
    task = args.task

    if task == "filter_corpus":
        # Task 1: output should be under Task_1_output/emotion/
        output_folder_path = output_folder_path / emotion
        filter_corpus(emotion, input_folder_path, output_folder_path)

    elif task == "json_to_csv":
        # Task 2: input is Task_1_output/emotion/, output is Task_2_output/
        json_to_csv(emotion, input_folder_path, output_folder_path)

if __name__ == "__main__":
    main()
