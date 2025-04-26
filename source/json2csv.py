import json
import pandas as pd
from pathlib import Path
from dataclasses import fields
from webcrawl import Article
from argparse import ArgumentParser

def read_json(path:Path):
    """
    Transform a json files into a single csv file where each raw is a json file
    """

    data = pd.DataFrame(columns=["excerpt","emotions"])
    if path.is_dir():
        for file in path.iterdir():

            if file.suffix == ".json":
                # current json file
                with open(file,'r',encoding='utf-8') as f:
                      json_file = json.load(f)
                      excerpts = []
                      for article in json_file["articles"]:
                          excerpt = article["excerpt"]
                          if isinstance(excerpt, list):
                              excerpt = " ".join(str(x) for x in excerpt)
                          excerpts.append(excerpt)
                               
                row = {
                    "excerpt": " ".join(excerpts),
                    "emotions": article["emotions"][0]
                    }
                data.loc[len(data)] = row
            elif file.is_dir():
                raise FileNotFoundError("no dir is accepted in dir")
            
    elif path.is_file and path.suffix == ".json":
        with open(path,'r',encoding='utf-8') as f:
            json_file = json.load(f)
            excerpts = ""
            for article in json_file["articles"]:
                excerpt = article["excerpt"]
                if isinstance(excerpt, list):
                    excerpt = " ".join(str(x) for x in excerpt)
                excerpts.append(excerpt)

        row = {
            "excerpt" : " ".join(excerpts),
            "emotions" : article["emotions"][0]
        }
        data.loc[len(data)] = row
    
    return data

def get_args():
    arg = ArgumentParser()
    arg.add_argument("-i","--input", help="input of cleanned data")
    arg.add_argument("-o","--output", help="csv result")
    return arg.parse_args()



def main():
    try:
        args = get_args()
    except:
        raise ValueError("to use python json2csv -i inputpath -o output")
    data = read_json(Path(args.input))
    data.to_csv(args.output,index=False)

        
if __name__ == "__main__":
    main()

