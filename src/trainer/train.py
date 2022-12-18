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

from dataset import ProposalDataset
from models import ConvolutionModel
from trainer import ModelTrainer


def run_exp(config):
    seed_everything(config.seed)
    exp_name = config.name + "_{}_{}".format(config.train.alpha, config.model)
    debug = config.debug
    batch_size = config.train.batch_size
    data_path = config.data.path
    df = pd.read_csv(data_path)
    df["text"] = [ast.literal_eval(d) for d in df["text"]]
    train_df, validate = train_test_split(df, test_size=0.2, tratify=df["attribute_id"])
    validate, test = train_test_split(
        validate, test_size=0.5, tratify=validate["attribute_id"]
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
        eval(config, test, mlflow_logger, test_dataloader)


def train(config, logger, train_dataloader, valid_dataloader):
    gpu_num = torch.cuda.device_count()
    save_path = "./model/baseline/model_{}_alpha_{}_seed_{}.pth".format(
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


def eval():
    pass
