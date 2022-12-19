import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import F1Score
from torchmetrics.functional import precision_recall
from transformers import RobertaConfig, RobertaModel

torch.autograd.set_detect_anomaly(True)


class ModelTrainer(pl.LightningModule):
    def __init__(self, alpha, model, save_path, learning_rate=1e-5):
        super().__init__()
        self.alpha = alpha
        self.model = model
        self.softmax = torch.nn.Softmax(dim=1)
        self.params = self.model.parameters()
        self.f1 = F1Score()
        self.lr = learning_rate
        self.path = save_path
        self.min_valid_loss = 1e1000

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, _):
        input_ids = batch["tokens"]
        attention_mask = batch["attention_mask"]
        crowd_dicision = batch["crowd_dicision"]
        system_dicision = batch["system_dicision"]
        system_out = batch["system_out"]
        annotator = batch["correct"]
        out = self.forward(input_ids, attention_mask)
        loss = self.loss_function(
            out, system_out, system_dicision, crowd_dicision, annotator
        )
        self.log_dict({"train_loss": loss}, on_epoch=True, on_step=True, logger=True)
        model_ans, _, _, _ = self.model.predict(
            out, system_out, system_dicision, crowd_dicision
        )
        acc, precision, recall, f1 = self.calc_all_metrics(model_ans, annotator)
        log_data = {
            "train_accuracy": acc,
            "train_precision": precision,
            "train_recall": recall,
            "train_f1": f1,
        }
        self.log_dict(log_data, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, _):
        input_ids = batch["tokens"]
        attention_mask = batch["attention_mask"]
        system_dicision = batch["system_dicision"]
        system_out = batch["system_out"]
        crowd_dicision = batch["crowd_dicision"]
        annotator = batch["correct"]
        out = self.forward(input_ids, attention_mask)

        loss = self.loss_function(
            out, system_out, system_dicision, crowd_dicision, annotator
        ).item()

        model_ans, s_count, c_count, _ = self.model.predict(
            out, system_out, system_dicision, crowd_dicision
        )
        acc, precision, recall, f1 = self.calc_all_metrics(model_ans, annotator)
        log_data = {
            "valid_accuracy": acc,
            "valid_precision": precision,
            "valid_recall": recall,
            "valid_f1": f1,
            "validation_loss": loss,
            "system_count": s_count,
            "crowd_count": c_count,
        }
        self.log_dict(log_data, on_epoch=True, logger=True)
        return log_data

    def validation_epoch_end(self, validation_epoch_outputs):
        system_all_count, crowd_all_count, loss = 0, 0, 0
        for out in validation_epoch_outputs:
            system_all_count += out["system_count"]
            crowd_all_count += out["crowd_count"]
            loss += out["validation_loss"]
        loss /= len(validation_epoch_outputs)
        data = {"system_count": system_all_count, "crowd_count": crowd_all_count}
        if loss <= self.min_valid_loss:
            torch.save(self.model.state_dict(), self.path)
        self.log_dict(data)

    def test_step(self, batch, _):
        input_ids = batch["tokens"]
        attention_mask = batch["attention_mask"]
        system_dicision = batch["system_dicision"]
        system_out = batch["system_out"]
        crowd_dicision = batch["crowd_dicision"]
        annotator = batch["correct"]
        start_idx = batch["start_idx"]
        end_idx = batch["end_idx"]
        out = self.forward(input_ids, attention_mask, start_idx, end_idx)
        if out is not None:
            model_ans, s_count, c_count = self.model.predict(
                out, system_out, system_dicision, crowd_dicision
            )
        else:
            model_ans, s_count, c_count = self.model.predict(
                system_dicision, crowd_dicision
            )
        acc, precision, recall, f1 = self.calc_all_metrics(model_ans, annotator)
        log_data = {
            "test_accuracy": acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1.item(),
            "system_count": s_count,
            "crowd_count": c_count,
        }
        return log_data

    def test_epoch_end(self, output_results):
        size = len(output_results)
        acc, precision, recall, f1, s_count, c_count = 0, 0, 0, 0, 0, 0
        for out in output_results:
            acc += out["test_accuracy"]
            precision += out["test_precision"]
            recall += out["test_recall"]
            f1 += out["test_f1"]
            s_count += out["system_count"]
            c_count += out["crowd_count"]
        acc /= size
        precision /= size
        recall /= size
        f1 /= size
        self.logger.log_metrics({"test_accuracy": acc})
        self.logger.log_metrics({"test_precision": precision})
        self.logger.log_metrics({"test_recall": recall})
        self.logger.log_metrics({"test_f1": f1})
        self.logger.log_metrics({"test_system_count": s_count})
        self.logger.log_metrics({"test_crowd_count": c_count})

    def predict_step(self, batch, _):
        input_ids = batch["tokens"]
        attention_mask = batch["attention_mask"]
        system_dicision = batch["system_dicision"]
        system_out = batch["system_out"]
        crowd_dicision = batch["crowd_dicision"]
        annotator = batch["correct"]
        start_idx = batch["start_idx"]
        end_idx = batch["end_idx"]
        out = self.forward(input_ids, attention_mask, start_idx, end_idx)
        if out is not None:
            model_ans, s_count, c_count = self.model.predict(
                out, system_out, system_dicision, crowd_dicision
            )
        else:
            model_ans, s_count, c_count = self.model.predict(
                system_dicision, crowd_dicision
            )
        acc, precision, recall, f1 = self.calc_all_metrics(model_ans, annotator)
        log_data = {
            "test_accuracy": acc,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1.item(),
            "system_count": s_count,
            "crowd_count": c_count,
        }
        return log_data

    def calc_all_metrics(self, model_ans, annotator):
        answer = annotator.to("cpu")
        acc, precision, recall, f1 = self.calc_metrics(answer, model_ans)
        return acc, precision, recall, f1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.get_params(), lr=self.lr)
        return optimizer

    # TODO:関数を変える
    def loss_function(
        self, output, system_out, system_dicision, crowd_dicision, annotator
    ):
        # log2(0)が入るのを防ぐために、微小値を足しておく
        output = torch.stack((system_out, output[:, 1]), -1)
        out = self.softmax(output) + 1e-10
        # import ipdb;ipdb.set_trace()
        # m1 = (cloud_dicision == annotator).to(int)
        m1 = (system_dicision != annotator).to(int)
        m2 = (system_dicision == crowd_dicision == annotator).to(int)
        loss = -(self.alpha * m2 + (1 - m1)) * torch.log2(out[:, 0]) - m1 * torch.log2(
            out[:, 1]
        )
        assert not torch.isnan(loss).any()
        loss = torch.mean(loss)
        return loss

    def calc_metrics(self, answer, result):
        acc = sum(answer == result) / len(answer)
        precision, recall = precision_recall(result, answer)
        acc = acc.item()
        precision = precision.item()
        recall = recall.item()
        f1 = self.f1(result, answer)
        return (acc, precision, recall, f1)
