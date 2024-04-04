#단순 선형 회귀: 넘파이
import numpy as np

x = np.array([
    [1],[2],[3],[4],[5],[6],[7],[8],[9],[10],
    [11],[12],[13],[14],[15],[16],[17],[18],[19],[20],
    [21],[22],[23],[24],[25],[26],[27],[28],[29],[30]
]) #(30,1) :2차원 행렬

#참고
np.array([1,2,3,4,5,6,7,8,9,10,
          11,12,13,14,15,16,17,18,19,20,
          21,22,23,24,25,26,27,28,29,30]) #(30,) : 1차원 벡터

np.array([[1,2,3,4,5,6,7,8,9,10,
          11,12,13,14,15,16,17,18,19,20,
          21,22,23,24,25,26,27,28,29,30]]) #(1, 30) :2차원 행렬

y = np.array([
    [0.94],[1.98],[2.88],[3.92],[3.96],[4.55],[5.64],[6.3],[7.44],[9.1],
    [8.46],[9.5],[10.67],[11.16],[14],[11.83],[14.4],[14.25],[16.2],[16.32],
    [17.46],[19.8],[18],[21.34],[22],[22.5],[24.57],[26.04],[21.6],[28.8]
])

weight = 0.0 #가중치
bias = 0.0 #편향
learning_rate = 0.005 #학습률

# Epoch : 모델 연산을 전체 데이터세트가 1회 통과하는 것 
for epoch in range(10000):
    y_hat = weight*x + bias #가설
    cost = ((y - y_hat)**2).mean() #손실 함수

    #가중치와 편향 갱신
    weight = weight - learning_rate*((y_hat-y)*x).mean()
    bias = bias - learning_rate*(y_hat-y).mean()

    #학습 기록 출력
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch:{epoch+1:4d},Weight:{weight:.3f},Bias:{bias:.3f},Cost:{cost:.3f}")

#%%
#단순 선형 회귀: 파이토치
import torch
from torch import optim


x = torch.FloatTensor([
    [1],[2],[3],[4],[5],[6],[7],[8],[9],[10],
    [11],[12],[13],[14],[15],[16],[17],[18],[19],[20],
    [21],[22],[23],[24],[25],[26],[27],[28],[29],[30]
]) #(30,1) :2차원 행렬


y = torch.FloatTensor([
    [0.94],[1.98],[2.88],[3.92],[3.96],[4.55],[5.64],[6.3],[7.44],[9.1],
    [8.46],[9.5],[10.67],[11.16],[14],[11.83],[14.4],[14.25],[16.2],[16.32],
    [17.46],[19.8],[18],[21.34],[22],[22.5],[24.57],[26.04],[21.6],[28.8]
])

weight = torch.zeros(1,requires_grad=True) #자동 미분 기능 사용 여부
bias = torch.zeros(1,requires_grad=True)
learning_rate=0.001

optimizer = optim.SGD([weight,bias],lr=learning_rate)

for epoch in range(10000):
    hypothesis = weight*x + bias
    cost = torch.mean((hypothesis - y)**2)

    #이건 뭘까
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if (epoch +1)%1000 == 0:
        print(f"Epoch:{epoch+1:4d},Weight:{weight.item():.3f},Bias:{bias.item():.3f},Cost:{cost:.3f}") 