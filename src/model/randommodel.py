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
        while len(ann_i) != annotator_count:
            sample = random.sample(range(len(crowd_dicision)))
            for s in sample:
                if s in crowd_i:
                    continue
                ann_i.append(s)
        ann_i = set(ann_i)
        c_counts, a_counts = 0, 0
        for i in range(len(crowd_dicision)):
            if i in crowd_i:
                model_ans.append(crowd_dicision[i])
                c_counts += 1
            elif i in ann_i:
                model_ans.append(anotator[i])
                a_counts += 1
            else:
                model_ans.append(system_dicision[i])
        assert c_counts == crowd_count
        assert a_counts == annotator_count
        return model_ans
