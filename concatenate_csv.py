import pandas as pd
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()

def concatanate_csv(args):
    """
    Task 3: Concatenate the csv files
    """

    if len(args.input) == 4:
        d1 = pd.read_csv(args.input[0])
        d2 = pd.read_csv(args.input[1])
        d3 = pd.read_csv(args.input[2])
        d4 = pd.read_csv(args.input[3])
        print("d1 (disgust) columns:", d1.columns.tolist())
        print("d2 (love) columns:", d2.columns.tolist())
        print("d3 (fear) columns:", d3.columns.tolist())
        print("d4 (surprise) columns:", d4.columns.tolist())
        # concatenate DataFrame
        combined_df = pd.concat([d1, d2, d3, d4], ignore_index=True)
        combined_df = combined_df.rename(columns={'text':'excerpt', 'emotion':'emotions'})
        # save concatenated df
        combined_df.to_csv(args.output, index=False)
        love = len(combined_df[combined_df['emotions'] == 'love'])
        disgust = len(combined_df[combined_df['emotions'] == 'disgust'])
        fear = len(combined_df[combined_df['emotions'] == 'fear'])
        surprise = len(combined_df[combined_df['emotions'] == 'surprise'])
        print(f"for each emotion there are {love} love, {disgust} disgust, {fear} fear, {surprise} surprise")
    else:
        raise ValueError("Excepts 4 csv files")
    

if __name__ == "__main__":
    args = get_args()
    concatanate_csv(args)




