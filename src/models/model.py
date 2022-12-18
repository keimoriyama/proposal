import torch.nn as nn
import torch

from transformers import RobertaConfig, RobertaModel
import math


class ConvolutionModel(nn.Module):
    def __init__(
        self,
        token_len,
        hidden_dim,
        out_dim,
        dropout_rate,
        kernel_size,
        stride,
        load_bert=False,
    ) -> None:
        super().__init__()
        self.token_len = token_len
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        self.config = RobertaConfig.from_pretrained("./model/config.json")
        self.bert = RobertaModel(config=self.config)
        if load_bert:
            self.bert.load_state_dict(torch.load("./model/bert_model.pth"))
        self.kernel_size = kernel_size
        self.stride = stride
        self.Conv1d = nn.Conv1d(
            self.hidden_dim,
            self.hidden_dim // 2,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        self.ConvOut = math.ceil(
            ((self.config.hidden_size + 2 * 0 - (self.kernel_size - 1) - 1) + 1)
            / self.stride
        )
        self.ReLU = nn.ReLU()
        self.Linear1 = nn.Linear(
            self.ConvOut * self.hidden_dim // 2, self.hidden_dim * 2
        )
        self.Linear2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.Linear3 = nn.Linear(self.hidden_dim, self.out_dim)
        self.params = (
            list(self.Linear1.parameters())
            + list(self.Linear2.parameters())
            + list(self.Linear3.parameters())
            + list(self.Conv1d.parameters())
        )

    def model(self, input):
        batch_size = input.size(0)
        out = self.ReLU(self.Conv1d(input))
        out = out.reshape(-1, self.hidden_dim // 2 * self.ConvOut)
        out = self.ReLU(self.Linear1(out))
        out = self.ReLU(self.Linear2(out))
        out = self.Linear3(out)
        return out

    def forward(self, input_ids, attention_mask, start_index=-1, end_index=-1):
        out = self.bert(input_ids, attention_mask=attention_mask)
        out = out["last_hidden_state"]
        out = self.model(out)
        return out

    def predict(self, out, system_out, system_dicision, crowd_dicision):
        model_ans = []
        system_crowd = []
        s_count, c_count = 0, 0
        for i, (s_out, c_out) in enumerate(zip(system_out, out[:, 1])):
            s_out = s_out.item()
            c_out = c_out.item()
            if s_out > c_out:
                model_ans.append(system_dicision[i])
                s_count += 1
                system_crowd.append("system")
            else:
                model_ans.append(crowd_dicision[i])
                c_count += 1
                system_crowd.append("crowd")
        model_ans = torch.Tensor(model_ans)
        return model_ans, s_count, c_count, system_crowd

    def get_params(self):
        return self.params
