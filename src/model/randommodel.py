import random

import torch
import torch.nn as nn

from model.modelinterface import ModelInterface


class RandomModel(ModelInterface):
    def __init__(self, out_dim) -> None:
        super().__init__()
        self.out_dim = out_dim

    # TODO:　annotatorの解答も引数に加える
    @classmethod
    def predict(
        cls, system_dicision, crowd_dicision, anotator, crowd_count, annotator_count
    ):
        model_ans = []
        crowd_i = set(random.sample(range(len(crowd_dicision)), crowd_count))
        ann_i = []
        while len(ann_i) == annotator_count:
            sample = list(random.sample(range(len(crowd_dicision)), annotator_count))
            for s in sample:
                if s in crowd_i:
                    continue
                ann_i.append(s)
        ann_i = set(ann_i)
        counts = 0
        for i in range(len(crowd_dicision)):
            if i in crowd_i:
                model_ans.append(crowd_dicision[i])
                counts += 1
            else:
                model_ans.append(system_dicision[i])
        assert counts == crowd_count
        return model_ans
