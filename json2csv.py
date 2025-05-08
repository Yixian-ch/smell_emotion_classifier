import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import fields
from webcrawl import Article, Corpus, load_json
from argparse import ArgumentParser

def create_directory() -> Path:
    """
    Create a directory for storing intermediate data
    
    Returns:
        Path: Path to the results directory
    """
    dir = Path("intermediate_data")
    dir.mkdir(exist_ok=True)

    return dir

def get_args() -> ArgumentParser:
    """
    Parse command line arguments.
    
    Returns:
        ArgumentParser: Parsed command line arguments
        
    Example:
        python json2csv.py -i ../data/disgust_output/data/ ../data/love_output/data/ ../data/fear_output/data/ -e disgust love fear -o ../test.csv

        Emotions must be given as the ordre of input files
    """
    arg: ArgumentParser = ArgumentParser(description="Filtering and reformatting data")
    arg.add_argument("-i", "--input_folders", nargs="+", required=True, help="input of web crawlled data, normally be a list of directories")
    arg.add_argument("-e", "--emotions", required=True, nargs="+",help="emotion to save for each given folder")
    arg.add_argument("--intermedia", default=False, help="if save the intermediate data")
    arg.add_argument("-o", "--output_file", required=True, help="csv result")
    return arg.parse_args()
from typing import List

def select_emotion(folders: List, emotions: List) -> Corpus:
    """
    Filters articles from a list of folders, keeping only those that are labeled 
    with a single, specified emotion.

    Each folder corresponds to a specific emotion. For each JSON file in the folder,
    the function loads its contents and selects articles that are labeled with the 
    target emotion and only that emotion.

    Args:
        folders (List): A list of folder paths containing JSON files. 
                        Each folder should contain data corresponding to a single emotion.
                        Example: ["fear_output/data/", "love_output/data/"]
        emotions (List): A list of target emotions corresponding to each folder.
                         The order of emotions must match the order of folders.

    Returns:
        Corpus: A Corpus object containing only the filtered articles that match 
                the specified criteria.
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
            "emotions": emotion
        }

        rows.append(row)
    
    return pd.DataFrame(rows)

def main() -> None:
    """
    Main function to process JSON files and convert them to a single CSV.
    """

    try:
        args = get_args()
        folders = Path(args.input_folders)
        emotions = args.emotions
    except:
        raise ValueError("to use python json2csv -i ../data/disgust_output/data/ ../data/love_output/data/ ../data/fear_output/data/ -e disgust love fear -o output.csv")
    

    print("Loading data...")
    filtered_corpus = select_emotion(folders, emotions)
    if args.intermedia:
        dir = create_directory()
        json.dump(filtered_corpus,dir,ensure_ascii=None,indent=2)

    data: pd.DataFrame = corpus_to_df(filtered_corpus)
    data.to_csv(args.output, index=False)
    print(f"\nSuccessfully created CSV with {len(data)} entries at {args.output}")

        
if __name__ == "__main__":
    main()