# 모델이 특정 피처, 패턴에 너무 많은 비중을 할당하지 않도록, 손실 함수에 규제를 가해 
# 모델의 일반화 성능을 향상

# 모델이 비교적 복잡, 데이터의 수가 적을때 활용 
# (모델이 단순하면 파라미터가 적어 정규화 필요없고, 데이터가 많거나 깨끗하면 x)

"""
LASSO (L1 정칙화)
"""
for x, y in train_dataloader :
   x = x.to(device)
   y = y.to(device)

   output = model(x)

   _lambda = 0.5
   l1_loss = sum(p.abs().sum() for p in model.parameters())

   loss = criterion(output, y) + + _lambda * l1_loss 

#간단해 보이지만 계산복잡도, 리소스 사용 높임
#%%
"""
Ridge (L2 정칙화)
"""
#하나의 특징이 너무 중요한 요소가 되지 않도록 규제를 가하는 것에 의미를 둔다.
for x, y in train_dataloader :
   x = x.to(device)
   y = y.to(device)

   output = model(x)

   _lambda = 0.5
   l2_loss = sum(p.pow(2.0).sum() for p in model.parameters())

   loss = criterion(output, y) + + _lambda * l2_loss

#%%
"""
가중치 감쇠
""" 
#L2 정규화와 동의어로 사용되지만, 가중치 감쇠는 손실함수에 규제 항을 추가하는 기술 자체를 의미

# 가중치 감쇠 적용 방식
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.01)
#파이토치에서는 L2 정규화와 동일한 의미
#%%

"""
엘라스틱 넷
"""
#L1 + L2
#트레이드 오프 문제에 유연, 더 많은 리소스 소모

"""
드롭아웃
"""
# 모델의 훈련 과정에서 일부 노드를 일정 비율로 제거하거나 0으로 설정해 과대적합을 방지
# 훈련 시간이 늘어남, 모든 노드를 사용해 학습하지 않으므로 데이터가 많아야 함
# 충분한 데이터세트, 비교적 깊은 모델

from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.layer2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

# 드롭아웃은 배치 정규화와 동시에 사용하지 않음
# 드롭아웃 -> 배치 정규화 순 /  추론 시에는 모든 노드를 사용하여 예측    
