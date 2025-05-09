from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from spacy.lang.fr.stop_words import STOP_WORDS as spacy_french_stopwords
import pandas as pd
import argparse
from pathlib import Path

def get_args():
    """
    CLI:
    python vectorizer_lab.py -ip <CSV file> -nd <number of documents> -sw <stopword library or "none">
    Note: min_df is set to 2. 
    """
    parser = argparse.ArgumentParser(description='Compare CountVectorizer and TfidfVectorizer on a French corpus')
    parser.add_argument("-nd", "--nb_docs", required=True, type=int)
    parser.add_argument("-ip", "--input_path", required=True, help="CSV file with a 'text' column.")
    parser.add_argument("-sw", "--stop_words_lib", choices=("nltk", "spacy", "combined", "none"), help="Choose stopword source", required=True)
    return parser.parse_args()

def run_vectorizers(path: Path, nb_docs: int, stop_words_lib: str) -> None:
    """
    This script applies both CountVectorizer and TfidfVectorizer to help evaluate 
    which tokenizer works best for a given French corpus. It also lets you compare 
    different stopword sources (or none). The goal is to make intentional choices that improve 
    representation and reduce the overall number of features.
    """
    df = pd.read_csv(path, encoding='utf-8')
    docs = df["text"].head(nb_docs).str.lower()

    if stop_words_lib == "spacy":
        stop_words = list(spacy_french_stopwords) +['neuf', 'qu', 'quelqu','bien', 'amour', 'eft']
    elif stop_words_lib == "nltk":
        stop_words = stopwords.words('french')
    elif stop_words_lib == "combined":
        stop_words = list(set(word.lower() for word in stopwords.words('french')).union(
                          word.lower() for word in spacy_french_stopwords))
    else:
        stop_words = None

    vectorizers = {
        "CountVectorizer": CountVectorizer(stop_words=stop_words, min_df=2),
        "TfidfVectorizer": TfidfVectorizer(stop_words=stop_words, min_df=2)
    }

    for name, vectorizer in vectorizers.items():
        print(f"\nVectorizer: {name}")

        matrix = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()

        df_vector = pd.DataFrame(matrix.toarray(), columns=feature_names)

        top_n = 20
        total_scores = df_vector.sum(axis=0)
        top_words = total_scores.sort_values(ascending=False).head(top_n)
        print(f"Stop words library: {args.stop_words_lib.upper()}")
        print(f"Data name: {Path(args.input_path).name} ")
        print(f"Top {top_n} words for {nb_docs} docs:")
        print(f"Total number of features for {name}: {len(feature_names)}\n")
        print(top_words)

if __name__ == "__main__":
    args = get_args()
    run_vectorizers(Path(args.input_path), args.nb_docs, args.stop_words_lib)