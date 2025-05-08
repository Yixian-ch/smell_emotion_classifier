import json
import pandas as pd
from pathlib import Path
from webcrawl import Corpus, load_json
import argparse
from typing import List, Dict

def create_directory() -> Path:
    """
    Create a directory for storing intermediate data
    
    Returns:
        Path: Path to the results directory
    """
    dir = Path("intermediate_data")
    dir.mkdir(exist_ok=True)

    return dir


def get_args():
    """
    Parse command line arguments.
    Allows iteration through one or several folder-and-emotion pairs.
    
    Returns:
        Namespace: Parsed command line arguments.
        
    CLI Usage:
        python json2csv.py -i <folder_path(s)> -e <emotion(s)> -o <output CSV file name>
    
    Example:
        python json2csv.py -i ../data/disgust_output/data/ ../data/love_output/data/ ../data/fear_output/data/ -e disgust love fear -o ../test.csv

    Note:
        Emotions must be provided in the same order as the input folders.
    """
    parser = argparse.ArgumentParser(description="Filter and reformat data.")
    parser.add_argument("-i", "--input_folders", nargs="+", required=True,
                        help="Input directories containing web-crawled data. Must be a list of directories.")
    parser.add_argument("-e", "--emotions", nargs="+", required=True,
                        help="List of emotions corresponding to each input folder.")
    parser.add_argument("--intermediate", default=False, 
                        help="If set, saves the intermediate data.")
    parser.add_argument("-o", "--output_file", required=True,
                        help="Path to the output CSV file.")

    args = parser.parse_args()

    if len(args.input_folders) != len(args.emotions):
        print("The number of input folders must match the number of emotions.")
        parser.print_help()
        exit(1)

    return args


def select_emotion(folders: List[Path], emotions: List[str]) -> Corpus:
    """
    Filters articles from a list of folders, keeping only those that are labeled 
    with a single, specified emotion.

    Each folder corresponds to a specific emotion. For each JSON file in the folder,
    the function loads its contents and selects articles that are labeled with the 
    target emotion and only that emotion.

    Args:
        folders (List[Path]): A list of folder paths containing JSON files. 
                        Each folder should contain data corresponding to a single emotion.
                        Example: ["fear_output/data/", "love_output/data/"]
        emotions (List[str]): A list of target emotions corresponding to each folder.
                         The order of emotions must match the order of folders.

    Returns:
        filtered_corpus (Corpus): A Corpus object containing only the filtered articles that match 
                the emotion or emotions.
    """
    filtered_corpus = Corpus()
    for idx, folder in enumerate(folders):
        print(f"Current folder: {folder.name}, target emotion: {emotions[idx]}")
        json_files = list(folder.rglob("*json"))
        for file in json_files:
            corpus = load_json(file)
            for article in corpus.articles:
                if emotions[idx] in article.emotions and len(article.emotions) == 1:
                    filtered_corpus.articles.append(article)

    print(f"Total articles after selection: {len(filtered_corpus.articles)}")
    return filtered_corpus

def corpus_to_df(corpus: Corpus) -> pd.DataFrame:
    """
    Convert a Corpus to a DataFrame for CSV export.
    
    Args:
        corpus (Corpus): A Corpus object containing Article objects
        
    Returns:
        pd.DataFrame: DataFrame with excerpt and emotions columns
    """
    rows: List[Dict[str, str]] = []

    for article in corpus.articles:
        # Handle case where emotions might be empty
        emotion: str = article.emotions[0] if article.emotions else ""

        row: Dict[str, str] = {
            "excerpt": article.excerpt,
            "emotion": emotion
        }

        rows.append(row)
    
    return pd.DataFrame(rows)

def main() -> None:
    """
    Main function to process JSON files and convert them to a single CSV.
    """
    args = get_args()
    folders = [Path(f) for f in args.input_folders]
    emotions = args.emotions
 
    print("Loading data...")
    filtered_corpus = select_emotion(folders, emotions)
    if args.intermediate:
        dir = create_directory()
        json.dump(filtered_corpus,dir,ensure_ascii=None,indent=2)

    data: pd.DataFrame = corpus_to_df(filtered_corpus)
    data.to_csv(args.output_file, index=False)
    print(f"\nSuccessfully created CSV with {data.shape[0]} entries at {args.output_file}")

        
if __name__ == "__main__":
    main()