import torch
import torch.nn as nn

def test_accumulation():
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float32)
    print(s)
    s = torch.tensor(0,dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01,dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)
    
class ToyModel(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        print("output of fc1.dtype",x.dtype)
        x = self.relu(x)
        x = self.ln(x)
        print("output of ln.dtype",x.dtype)
        x = self.fc2(x)
        print("the logits.dtype",x.dtype)
        return x
def test_autocast(data_type):
    model = ToyModel(10,10).cuda()
    with torch.autocast(device_type="cuda",dtype=data_type):
        x = torch.randn(10,10).cuda()
        label = torch.randint(0,10,(10,)).cuda()
        y = model(x)
        loss = torch.nn.functional.cross_entropy(y,label)
        loss.backward()
        print("loss.dtype",loss.dtype)
        print("model.fc1.weight.dtype",model.fc1.weight.dtype) ### weight存储类型不变，中间计算精度可能会改变为bf16
        print("model.fc2.weight.dtype",model.fc2.weight.dtype)
        print("model.ln.weight.dtype",model.ln.weight.dtype)
        print("model.ln.bias.dtype",model.ln.bias.dtype)
        print("model gradient dt",model.fc1.weight.grad.dtype)
        
if __name__ == "__main__":
    test_autocast(torch.bfloat16)
