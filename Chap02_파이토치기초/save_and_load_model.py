"""
모델 전체 저장/불러오기
"""
import torch
from torch import nn

# torch.save(model,path)
#모델 전체 저장이므로 작게는 몇 메가~ 기가바이트까지 용량 필요 

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load('../models/model.pt', map_location=device)
#모델을 불러올 시 동일한 구조의 클래스가 선언 되어 있어야 함/ 아니라면 AttributeError가 발생
#%%
#이런 식으로 클래스가 선언되어 있어야 함
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2,1)

    def forward(self,x):
        x = self.layer(x)
        return x

model = torch.load('../models/model.pt', map_location=device)
#%%
#모델 구조를 알 수 없다면? 모델 구조를 출력해 확인->CustomModel 클래스에 동일한 형태로 모델을 구현
class CustomModel(nn.Module):
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load('../models/model.pt', map_location=device)
print(model)
# CustomModel(
#   (layer): Linear(in_features=2, out_features=1, bias=True)
# )
# 이 때, 변수의 명칭까지 동일한 형태로 구현해야 함 

