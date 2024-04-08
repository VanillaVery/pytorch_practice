#데이터세트; 데이터의 집합 

#데이터는 파일 경로로 제공되거나, 전처리 단계가 필요한 경우도 있음
#데이터를 변형하고 매핑하는 코드를 학습과정에 직접 반영하면 
# 모듈화, 재사용성, 가독성을 떨어뜨리는 주요 원인이 됨

#이러한 현상을 방지, 코드를 구조적으로 설계할 수 있도록 
#데이터세트와 데이터로더를 사용 

#%%
"""
#데이터세트
#학습에 필요한 데이터 샘플을 정제하고 정답을 저장하는 기능 제공
"""
#데이터 세트 클래스의 기본형 
class Dataset:
    #입력된 데이터의 전처리 과정을 수행하는 메서드 
    def __init__(self,data,*arg,**kwargs):
        self.data=data
    
    #학습을 진행할 때 사용되는 하나의 행을 불러오는 과정
    def __getitem__(self,index):
        return tuple(data[index] for data in data.tensors)
    
    #학습에 사용된 전체 데이터세트의 개수 반환
    def __len__(self):
        return self.data[0].size(0)
    
#%%
"""
데이터로더
데이터세트에 저장된 데이터를 어떠한 방식으로 불러와 활용할지 정의
학습을 원활하게 진행할 수 있도록 배치크기, 셔플, num_workers등의 기능제공
"""
#%%
"""
다중 선형 회귀
"""
#기본 구조 선언
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

train_x = torch.FloatTensor([
    [1,2],[2,3],[3,4],[4,5],[5,6],[6,7]
])

train_y = torch.FloatTensor([
    [0.1,1.5],[1,2.8],[1.9,4.1],[2.8,5.4],[3.7,6.7],[4.6,8]
])

#%%
#데이터세트와 데이터로더
train_dataset = TensorDataset(train_x, train_y) #여러개의 값 입력 가능
train_dataloader = DataLoader(train_dataset, 
                              batch_size = 2, 
                              shuffle = True, 
                              drop_last = True) #배치 크기에 맞지 않는 배치 제거


model = nn.Linear(2,2,bias=True)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.001)

#%%
# 데이터로더 적용
for epoch in range(20000):
    cost = 0.0 #에포크마다 오차를 다시 계산해야 하기 때문에 초기화
    
    for batch in train_dataloader:
        x, y = batch
        output = model(x)

        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss # 오차에 손실을 누적
    
    cost = cost / len(train_dataloader) # 오차 평균

    if (epoch+1) % 1000 == 0:
        print(f"Epoch : {epoch+1:4d}, Model : {list(model.parameters())}, Cost : {cost:.3f}")

# 이렇게 데이터로더를 활용하면, 자여느럽게 배치 구조로 코드가 변경
# 이 경우 학습 데이터의 구조나 형태가 변경되더라도, 실제 학습에 사용되는 코드는 변경되지 않아서 각 모듈에 집중할 수 있음
