
import torch
import torch.nn as nn
from cs336_basics.optimizer import AdamW
from utils import FSDP_Optimizer

class toy_network(nn.Module):
    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)
        layers = []
        for i in range(3):
            layers.append(nn.Linear(128, 128))
            layers.append(nn.ReLU())
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)

if __name__ == "__main__":
    model = toy_network()
    fsdp = FSDP_Optimizer(model.parameters(), AdamW)
    