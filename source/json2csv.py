import json
import pandas as pd
from pathlib import Path
from dataclasses import fields
from webcrawl import Article

def read_json(path:Path):
    experts = []
    emotion = []
    if path.is_dir():
        for file in path.iterdir():
            if file.suffix == ".json":
                with open(file,'r',encoding='utf-8') as f:
                      data = json.load(f)
                      for article in data["articles"]:
                          experts.append(article["excerpt"])
                          emotion.append(article["emotions"])
            elif file.is_dir():
                read_json(file)
    elif path.is_file and path.suffix == ".json":
        with open(path,'r',encoding='utf-8') as f:
            data = json.load(f)
            for article in data["articles"]:
                experts.append(article["excerpt"])
                emotion.append(article["emotions"])
    
    return pd.DataFrame({"experts":experts,"emotion":emotion})


df = read_json(Path("/home/chen/Desktop/smell_emotion_classifier/love_cleaned"))
print(df.head())
        


