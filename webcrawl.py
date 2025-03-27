import requests
import json
from pathlib import Path
from dataclasses import dataclass,field
import re
import time
from argparse import ArgumentParser

@dataclass
class Smell:
    doc_id:str
    emotion:str
    text:str
    # title:str
    # source:str


# use a different IP to do webcrawl
url = "https://explorer.odeuropa.eu/api/search?filter_emotion=http%3A%2F%2Fdata.odeuropa.eu%2Fvocabulary%2Fplutchik%2Flove&hl=en&page=1&sort=&type=smells"

def get_json_data(api_url,out_put:Path)->None:
    out_put.mkdir(parents=True,exist_ok=True)
    for i in range(1,501):
        url = re.sub(r"page=\d+",f"page={i}",api_url)
        response = requests.get(url).json()
        file_path = out_put / f"page_{i}.json"
        with open(file_path,'w+') as f:
            json.dump(response,f,indent=2)

        # time.sleep(1)
        if i % 10 == 0:
            print(f"processed {i} pages")

# get_json_data(url,Path("data"))

def read_json(path:Path) ->list:
    with open(path,'r') as f:
        raw = json.load(f)['results']
        results = []

        for id, article in enumerate(raw,start=1):
            emotion = [i["label"] for i in article.get("emotion") if i["label"]=="love"][0]

            result = Smell(
                doc_id = id,
                emotion = emotion,
                text = article.get("text")
            )
            results.append(result)

        return results
            

def main():
    parser = ArgumentParser()
    
    parser.add_argument("url")
    parser.add_argument("output")



