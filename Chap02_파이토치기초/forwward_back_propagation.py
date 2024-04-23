# 순전파: 입력이 주어지면 신경망의 출력을 계산하는 프로세스
# 역전파: 순전파 과정을 통해 나온 오차를 활용하여 각 계층의 가중치, 편향을 최소화

#%%
#모델 구조와 초깃값
import torch
from torch import nn
from torch import optim

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(2,2),
            nn.Sigmoid()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(2,1),
            nn.Sigmoid()
        )

        self.layer1[0].weight.data = torch.nn.Parameter(
            torch.Tensor([[0.4352, 0.3545],
                          [0.1951,0.4835]])
        )

        
        self.layer1[0].bias.data = torch.nn.Parameter(
            torch.Tensor([-0.1419,0.0439])
        )

        self.layer2[0].weight.data = torch.nn.Parameter(
        torch.Tensor([[-0.1725, 0.1129]])
        )

        self.layer2[0].bias.data = torch.nn.Parameter(
        torch.Tensor([-0.3043])
        )

device = "cuda" if torch.cuda_is_available() else "cpu"
model = CustomModel().to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.SGD(model.parameters(),lr = 1)
