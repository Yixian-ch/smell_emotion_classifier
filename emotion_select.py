from webcrawl import Path,json,argparse

def clean(path:Path,emotion:str) -> None:
    emotions = ["trust","love","sadness","fear"]
    for file in path.iterdir():
        with open(file,"r+") as f:
            corpus = json.load(f)
        
        filtered_articles = [article for article in corpus["articles"] if len(article["emotions"])==1 and article["emotions"][0].lower()==emotion] 

        cleaned_data = {"articles": filtered_articles }
        with open(file,"w+") as f:
            json.dump(cleaned_data,f,ensure_ascii=False,indent=2)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path2clean", help="where the data are saved")
    parser.add_argument("emotion", help="the emotion to filter")
    args = parser.parse_args()

    clean(Path(args.path2clean),args.emotion)

if __name__ == "__main__":
    main()