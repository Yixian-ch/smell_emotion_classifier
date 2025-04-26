from webcrawl import Path,json,argparse,extract_data,save_json

def clean(path:Path,emotion:str,output:Path) -> None:
    emotions = ["trust","love","sadness","fear"]
    output.mkdir(parents=True,exist_ok=True)
    for file in path.iterdir():
        with open(file,"r+") as f:
            corpus = json.load(f)
        
        filtered_articles = [article for article in corpus["articles"] if len(article["emotions"])==1 and article["emotions"][0].lower()==emotion] 

        if len(filtered_articles) >= 1: # not empty file will be saved
            cleaned_data = {"articles": filtered_articles }
            output_file = output / file.name
            with open(output_file,"w+") as f:
                json.dump(cleaned_data,f,ensure_ascii=False,indent=2)

def restore(input_path:Path,output_path:Path): 
    """
    If you spoil your data, the way to recover them
    """
    output_path.mkdir(parent=True,exist_ok=True)
    for i, path in enumerate(input_path.iterdir()):
        if path.is_file and path.suffix == ".json":
            data = extract_data(path)
            save_path = output_path / f"data_page_{i+1}.json"
            save_json(data,save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path2clean", help="where the data is saved")
    parser.add_argument("emotion", help="the emotion to filter")
    parser.add_argument("output", help="where the filtered files will be saved")
    parser.add_argument("-r","--recover", nargs=2, metavar=('INPUT','OUTPUT'),help="the data are spoiled? No worries, you can recover them from raw data. to use: INPUT path(raw data) OUTPUT (restored data)")
    args = parser.parse_args()

    if args.recover:
        input,output = args.recover
        restore(Path(input),Path(output))
    clean(Path(args.path2clean),args.emotion,Path(args.output))

if __name__ == "__main__":
    main()