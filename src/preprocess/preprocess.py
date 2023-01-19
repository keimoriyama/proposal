import ipdb
import pandas as pd
from omegaconf import OmegaConf
from tokenizer import JanomeBpeTokenizer
import random

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

tokenizer = JanomeBpeTokenizer("../model/codecs.txt")

data_path = "./data/shinra2022jp_lake_system_crowdsourcing_884.csv"


def main():
    config = OmegaConf.load("./config/config.yml")
    path = "./data/system_df.csv"
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    df["index_id"] = [i + 1000 for i in range(len(df))]
    df["text_text"] = df["text_text"].apply(remove_return)
    df = df.filter(
        regex="index_id|system_*|correct|worker_*|text_text|attribute|page_id"
    )
    df = df.fillna(False)
    system = df.filter(regex="index_id|system_*")
    # 集計
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
    
    if config.dataset.name == "artificial":

        df["system_dicision"] = df["system_true_count"] > df["system_false_count"]
        df["crowd_dicision"] = df["crowd_true_count"] > df["crowd_false_count"]
        data = []
        for i in range(len(df)):
            d = df.iloc[i]
            di = {}
            if random.random() < 0.3:
                d['crowd_dicision'] = not(d['correct'])
                d['system_dicision'] = not(d['correct'])
            di = d.to_dict()
            data.append(di)
        df = pd.DataFrame(data)
    
    if config.dataset.name == "worse_system":
        # ワーカーのデータ作成
        dicision_df = system["system_true_count"] >= 3
        # import ipdb;ipdb.set_trace()
        column_name = "crowd_dicision"
        df[column_name] = dicision_df
        # システムのデータ作成
        # special_attribute = [a for i, a in enumerate(df['attribute'].value_counts().index) if i %2 == 0]
        special_attribute = ['動物', "別名"]
        datas = []
        i = -1
        for _, data in df.iterrows():
            # import ipdb; ipdb.set_trace()
            data["system_dicision"] = data["system_true_count"] >= 2
            if (data['attribute'] not in special_attribute
                and (data["system_dicision"] == data["correct"])
                and (random.uniform(0, 1) >= 0.5)
            ):
                data["system_dicision"] = not(data["correct"])

            datas.append(data)
        worker_df = pd.DataFrame(datas)
        # import ipdb; ipdb.set_trace()
        worker_df = worker_df[["index_id", "system_dicision", "crowd_dicision"]]
        df = pd.merge(df, worker_df)
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
    print("system")
    system_score = calc_metrics(df["correct"], df["system_dicision"])
    system_score['kind'] = 'system'
    print("crowd")
    crowd_score = calc_metrics(df["correct"], df["crowd_dicision"])
    crowd_score['kind'] = 'crowd'
    print("annotator")
    annotator_score = calc_metrics(df["correct"], df["correct"])
    annotator_score['kind'] = 'annotator'
    scores = pd.DataFrame([system_score, crowd_score, annotator_score])
    scores.to_csv("./output/only_scores.csv", index=False)
    print("kinds")
    scores = []
    for att in df['attribute'].unique():
        d = df[df['attribute'] == att]
        print("attribute: {} number of data: {}".format(att, len(d)))
        score = calc_metrics(d['correct'], d['system_dicision'])
        score['attribute'] = att
        score['data_num'] = len(d)
        scores.append(score)
    scores = pd.DataFrame(scores)
    scores.to_csv("./output/attribute_scores.csv", index=False)



def tokenize_text(text):
    return tokenizer.tokenize(text)[0]


def remove_return(s):
    return s.replace("\n", "")


def calc_metrics(ans, out):
    acc = accuracy_score(ans, out)
    pre = precision_score(ans, out, zero_division=0)
    recall = recall_score(ans, out)
    f1 = f1_score(ans, out)
    print(
        "accuracy: {:.3}, f1: {:.3}, precision: {:.3}, recall: {:.3}".format(
            acc, f1, pre, recall
        )
    )
    return {"accuracy":acc,"f1":f1,"recall":recall, "precisoin":pre}

if __name__ == "__main__":
    main()
