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
    arg.add_argument("-i", "--input", nargs="+", required=True, help="input of web crawlled data, normally be a list of directories")
    arg.add_argument("-e", "--emotions", required=True, nargs="+",help="emotion to save for each given folder")
    arg.add_argument("--intermedia", default=False, help="if save the intermediate data")
    arg.add_argument("-o", "--output", required=True, help="csv result")
    return arg.parse_args()

def select_emotion(args) -> Corpus:
    """
    Read several folders and return same number of folders of articles correspond to given emotion

    Returns:
        Corpus: selected corpus
    """
    filtered_corpus = Corpus()

    for idx, directory in enumerate(args.input):
        print(f"current folder {directory}, select emotion {args.emotions[idx]}")

        for file in Path(directory).iterdir():
            corpus = load_json(file)
            for article in corpus.articles:
                if args.emotions[idx] in article.emotions and len(article.emotions) == 1:
                    filtered_corpus.articles.append(article)

    print(f"total articles after the selection: {len(filtered_corpus.articles)}")
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
    except:
        raise ValueError("to use python json2csv -i ../data/disgust_output/data/ ../data/love_output/data/ ../data/fear_output/data/ -e disgust love fear -o output.csv")
    

    print("Loading data...")
    filtered_corpus = select_emotion(args)

    if args.intermedia:
        dir = create_directory()
        json.dump(filtered_corpus,dir,ensure_ascii=None,indent=2)

    data: pd.DataFrame = corpus_to_df(filtered_corpus)
    data.to_csv(args.output, index=False)
    print(f"\nSuccessfully created CSV with {len(data)} entries at {args.output}")

        
if __name__ == "__main__":
    main()