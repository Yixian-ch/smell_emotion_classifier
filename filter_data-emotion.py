from pathlib import Path
from webcrawl import load_json, save_json, Corpus

def filter_corpus(emotion:str, folder_path:Path, output_path:Path)-> None:
    """
    Filters out the specified emotion class.
    """
    paths = folder_path.rglob("*.json")

    emotion_counter = 0
    article_counter = 0
    filtered_corpus = Corpus()
    for file_path in paths:        
        corpus = load_json(file_path)
        articles = corpus.articles

        for article in articles:
            article_counter+=1
            if article.emotions == [f"{emotion}"]: 
                emotion_counter+=1
                filtered_corpus.articles.append(article)
    
    save_json(filtered_corpus, output_path)
    print(f"Total articles processed: {article_counter}")
    print(f"Articles with emotion '{emotion}': {emotion_counter}")

emotion = "fear"
data_path = Path("data/Odeuropa_output-test3-fear/data")
output_path = Path(f"data/filtered_data/filtered_{emotion}.json")
filtered_corpus = filter_corpus(emotion, data_path, output_path)
