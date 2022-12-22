import ipdb
import pandas as pd
from omegaconf import OmegaConf
from tokenizer import JanomeBpeTokenizer
import random

tokenizer = JanomeBpeTokenizer("../model/codecs.txt")

data_path = "./data/shinra2022jp_lake_system_crowdsourcing_884.csv"


def main():
    config = OmegaConf.load("./config/config.yml")
    df = pd.read_csv(data_path)
    df = df.reset_index(drop=True)
    df["index_id"] = [i + 1000 for i in range(len(df))]
    df["text_text"] = df["text_text"].apply(remove_return)
    df = df.filter(
        regex="index_id|system_*|correct|worker_*|text_text|attribute|page_id"
    )
    df = df.fillna(False)
    system = df.filter(regex="index_id|system_*")
    system["system_true_count"] = (system == True).sum(axis=1)
    system["system_false_count"] = (system == False).sum(axis=1)
    system["system_out"] = system["system_true_count"] / (
        system["system_true_count"] + system["system_false_count"]
    )

    crowd = df.filter(regex="index_id|worker_*")
    crowd["crowd_true_count"] = (crowd == "yes").sum(axis=1)
    crowd["crowd_false_count"] = 10 - crowd["crowd_true_count"]
    crowd["crowd_out"] = crowd["crowd_true_count"] / (
        crowd["crowd_true_count"] + crowd["crowd_false_count"]
    )

    df = pd.merge(df, system)
    df = pd.merge(df, crowd)
    df["text"] = df["text_text"].apply(tokenize_text)
    df["system_dicision"] = df["system_true_count"] > df["system_false_count"]
    df["crowd_dicision"] = df["crowd_true_count"] > df["crowd_false_count"]
    if config.dataset.name == "artificial":
        data = []
        for i in range(len(df)):
            d = df.iloc[i]
            di = {}
            if random.random() > 0.3:
                d['crowd_dicision'] = not(d['correct'])
                d['system_dicision'] = not(d['correct'])
            di = d.to_dict()
            data.append(di)
        df = pd.DataFrame(data)
    df = (
        df[
            [
                "page_id",
                "system_dicision",
                "crowd_dicision",
                "correct",
                "text",
                "attribute",
                "system_out",
                "crowd_out",
            ]
        ]
        .replace(True, 1)
        .replace(False, 0)
        .reset_index()
    )
    df.to_csv("./data/train_{}.csv".format("sample_"+config.dataset.name), index=False)


def tokenize_text(text):
    return tokenizer.tokenize(text)[0]


def remove_return(s):
    return s.replace("\n", "")


if __name__ == "__main__":
    main()
