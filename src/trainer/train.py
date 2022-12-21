import ast

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix, F1Score
from torchmetrics.functional import precision_recall
from tqdm import tqdm

from dataset import ProposalDataset
from model import ConvolutionModel, RandomModel
from trainer.trainer import ModelTrainer


def run_exp(config):
    seed_everything(config.seed)
    exp_name = config.name
    debug = config.debug
    batch_size = config.train.batch_size
    data_path = config.dataset.path
    df = pd.read_csv(data_path)
    df["text"] = [ast.literal_eval(d) for d in df["text"]]
    train_df, validate = train_test_split(df, test_size=0.2, stratify=df["correct"])
    validate, test = train_test_split(
        validate, test_size=0.5, stratify=validate["correct"]
    )
    if debug:
        train_df = train_df[: batch_size * 2]
        validate = validate[: batch_size * 2]
        config.train.epoch = 3
    train_df = train_df.reset_index()
    validate_df = validate.reset_index()
    test_df = test.reset_index()

    train_dataset = ProposalDataset(train_df)
    valid_dataset = ProposalDataset(validate_df)
    test_dataset = ProposalDataset(test_df)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=config.dataset.num_workers
    )
    validate_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=config.dataset.num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=config.dataset.num_workers
    )
    if config.mode == "train":
        mlflow_logger = MLFlowLogger(experiment_name=exp_name)
        mlflow_logger.log_hyperparams(config.train)
        # import ipdb;ipdb.set_trace()
        mlflow_logger.log_hyperparams({"mode": config.mode})
        mlflow_logger.log_hyperparams({"seed": config.seed})
        mlflow_logger.log_hyperparams({"model": config.model})
        train(config, mlflow_logger, train_dataloader, validate_dataloader)
    else:
        mlflow_logger = MLFlowLogger(experiment_name="test")
        mlflow_logger.log_hyperparams(config.train)
        mlflow_logger.log_hyperparams({"mode": config.mode})
        mlflow_logger.log_hyperparams({"seed": config.seed})
        mlflow_logger.log_hyperparams({"model": config.model})
        eval(
            config,
            test,
            test_dataloader,
            mlflow_logger,
        )


def train(config, logger, train_dataloader, valid_dataloader):
    gpu_num = torch.cuda.device_count()
    save_path = "./model/proposal/model_{}_alpha_{}_seed_{}.pth".format(
        config.model, config.train.alpha, config.seed
    )

    trainer = pl.Trainer(
        max_epochs=config.train.epoch,
        logger=logger,
        strategy="ddp",
        gpus=gpu_num,
    )
    model = ConvolutionModel(
        token_len=512,
        out_dim=config.train.out_dim,
        hidden_dim=config.train.hidden_dim,
        dropout_rate=config.train.dropout_rate,
        kernel_size=4,
        stride=2,
        load_bert=True,
    )

    modelTrainer = ModelTrainer(
        alpha=config.train.alpha,
        model=model,
        save_path=save_path,
        learning_rate=config.train.learning_rate,
    )
    trainer.fit(modelTrainer, train_dataloader, valid_dataloader)


def eval(config, test, test_dataloader, logger):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvolutionModel(
        token_len=512,
        out_dim=config.train.out_dim,
        hidden_dim=config.train.hidden_dim,
        dropout_rate=config.train.dropout_rate,
        kernel_size=4,
        stride=2,
        load_bert=False,
    )
    path = "./model/proposal/model_{}_alpha_{}_seed_{}.pth".format(
        config.model, config.train.alpha, config.seed
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    predictions = []
    data = []
    for batch in test_dataloader:
        input_ids = batch["tokens"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        system_dicision = batch["system_dicision"].to(device)
        system_out = batch["system_out"].to(device)
        crowd_dicision = batch["crowd_dicision"].to(device)
        annotator = batch["correct"].to(device)
        text = batch["text"]
        attribute = batch["attribute"]
        answer = annotator.to("cpu")
        out = model(input_ids, attention_mask)
        model_ans, s_count, c_count, a_count, method = model.predict(
            out, system_out, system_dicision, crowd_dicision, annotator
        )

        texts = []
        for i in range(len(text[0])):
            s = ""
            for t in text:
                if t[i] == "<s>":
                    continue
                elif t[i] == "</s>":
                    break
                s += t[i]
            texts.append(s)
        for t, m_a, m, att, ans in zip(texts, model_ans, method, attribute, answer):
            d = {
                "text": t,
                "attribute": att,
                "model answer": int(m_a.item()),
                "model choise": m,
                "answer": ans.item(),
            }
            data.append(d)

        acc, precision, recall, f1 = calc_metrics(answer, model_ans)
        predictions += [
            {
                "test_accuracy": acc,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1.item(),
                "system_count": s_count,
                "crowd_count": c_count,
                "annotator_count": a_count,
            }
        ]
    df = pd.DataFrame(data)
    # import ipdb;ipdb.set_trace()
    c_mat = confusion_matrix(df["answer"], df["model answer"])
    tn = c_mat[0][0]
    fn = c_mat[1][0]
    tp = c_mat[1][1]
    fp = c_mat[0][1]
    logger.log_metrics({"test true negative": tn})
    logger.log_metrics({"test false negative": fn})
    logger.log_metrics({"test true positive": tp})
    logger.log_metrics({"test false positive": fp})
    title = "result_model_{}_alpha_{}_seed_{}.csv".format(
        config.model, config.train.alpha, config.seed
    )
    df.to_csv("./output/" + title, index=False)
    eval_with_random(predictions, test, logger, config)


def calc_metrics(answer, result):
    f1_score = F1Score()
    acc = sum(answer == result) / len(answer)
    precision, recall = precision_recall(result, answer)
    acc = acc.item()
    precision = precision.item()
    recall = recall.item()
    f1 = f1_score(result, answer)
    return (acc, precision, recall, f1)


def eval_with_random(predictions, test, logger, config):
    size = len(predictions)
    crowd_d = test["crowd_dicision"].to_list()
    system_d = test["system_dicision"].to_list()
    answer = test["correct"].to_list()
    acc, precision, recall, f1, s_count, c_count, a_count = 0, 0, 0, 0, 0, 0, 0
    for out in predictions:
        acc += out["test_accuracy"]
        precision += out["test_precision"]
        recall += out["test_recall"]
        f1 += out["test_f1"]
        s_count += out["system_count"]
        c_count += out["crowd_count"]
        a_count += out["annotator_count"]
    acc /= size
    precision /= size
    recall /= size
    f1 /= size
    logger.log_metrics({"test_accuracy": acc})
    logger.log_metrics({"test_precision": precision})
    logger.log_metrics({"test_recall": recall})
    logger.log_metrics({"test_f1": f1})
    logger.log_metrics({"test_system_count": s_count})
    logger.log_metrics({"test_crowd_count": c_count})
    logger.log_metrics({"test_annotator_count": a_count})
    accs, precisions, recalls, f1s = [], [], [], []
    tns, tps, fns, fps, = (
        [],
        [],
        [],
        [],
    )
    # シード値かえて100かい回す
    for i in range(100):
        seed_everything(config.seed + i + 1)
        random_pred = RandomModel.predict(system_d, crowd_d, answer, c_count, a_count)
        acc = sum([a == r for a, r in zip(answer, random_pred)]) / len(answer)
        precision, recall, f1, _ = precision_recall_fscore_support(
            random_pred, answer, average="macro"
        )
        c_mat = confusion_matrix(answer, random_pred)
        tns.append(c_mat[0][0])
        fns.append(c_mat[1][0])
        tps.append(c_mat[1][1])
        fps.append(c_mat[0][1])
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    def calc_mean(l):
        return sum(l) / len(l)

    acc = calc_mean(accs)
    precision = calc_mean(precisions)
    recall = calc_mean(recalls)
    f1 = calc_mean(f1s)
    tn = calc_mean(tns)
    tp = calc_mean(tps)
    fn = calc_mean(fns)
    fp = calc_mean(fps)
    print(acc, precision, recall, f1)
    logger.log_metrics({"random_accuracy": acc})
    logger.log_metrics({"random_precision": precision})
    logger.log_metrics({"random_recall": recall})
    logger.log_metrics({"random_f1": f1})
    logger.log_metrics({"random true negative": tn})
    logger.log_metrics({"random false negative": fn})
    logger.log_metrics({"random true positive": tp})
    logger.log_metrics({"random false positive": fp})
