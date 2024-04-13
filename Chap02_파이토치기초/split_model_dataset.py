from torch import nn
#모델 구현
#신경망 패키지의 Module 클래스를 활용
#새로운 모델 클래스를 생성하려면 모듈 클래스를상속받아 임의의 서브 클래스 생성 

"""
모듈 클래스 
"""
# 모듈 클래스는 초기화 메서드와 순방향 메서드를 재정의하여 활용 

#모듈 클래스 기본형
class Model(nn.Module):
    def __init__(self):
        super().__init__() #계층 정의 전에, 모듈 클래스의 속성 초기화 
                            # 초기화 시 서브 클래스인 model에서 부모 클래스의 속성 사용 가능
        self.conv1 = nn.Conv2d(1,20,5)
        self.conv3 = nn.Conv2d(20,20,5)

    def forward(self, x): #모델 매개변수를 활용해 신경망 구조 설계
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x
    
    # 초기화 메서드에서 부모 클래스를 초기화했으므로, backward 연산은 정의하지 않아도 됨.
    # 파이토치 Autograd에서 모델의 매개변수를 역으로 전파해 자동으로 기울기 또는 변화도계산
    # 따라서 별도의 메서드로 역전파 기능을 구성하지 않아도 됨
#%%
"""
비선형 회귀
"""
import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset,DataLoader

#사용자 정의 데이터세트 
class CustomDataset(Dataset):
    def __init__(self,file_path):
        df = pd.read_csv(file_path)
        self.x = df.iloc[:,0].values
        self.y = df.iloc[:,1].values
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.FloatTensor([self.x[index] **2 , self.x[index]])
        y = torch.FloatTensor([self.y[index]])
        return x,y
    
    def __len__(self):
        return self.length
    

#사용자 정의 모델
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2,1)

    def forward(self,x):
        x = self.layer(x)
        return x
    

#사용자 정의 데이터세트 및 데이터로더
train_dataset = CustomDataset('../datasets/non_linear.csv')
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CustomModel().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(10000):
    cost = 0.0

    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss

    cost = cost / len(train_dataloader)

    if (epoch +1) % 1000 == 0:
        print(f"epoch : {epoch +1:4d}, Model: {list(model.parameters())}, Cost: {cost:.3f}")


 #%%
"""
 모델 평가
"""   

with torch.no_grad(): #자동 미분 기능을 사용하지 않도록 설정해 메모리 사용량을 줄임
    model.eval() # 평가 모드로 반환(특정 계층에서는 평가 모드와 학습 모드가 다름)
    inputs = torch.FloatTensor([
        [1**2,1],
        [5**2,5],
        [11**2,11]
    ]).to(device)

    outputs = model(inputs)
    print(outputs)

 #모델 저장
torch.save(
     model, "../models/model.pt"
 )

torch.save(
    model.state_dict(), "../models/model_state_dict.pt"
)
#%%
"""
데이터세트 분리
"""
import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset,DataLoader,random_split

#사용자 정의 데이터세트 
class CustomDataset(Dataset):
    def __init__(self,file_path):
        df = pd.read_csv(file_path)
        self.x = df.iloc[:,0].values
        self.y = df.iloc[:,1].values
        self.length = len(df)

    def __getitem__(self, index):
        x = torch.FloatTensor([self.x[index] **2 , self.x[index]])
        y = torch.FloatTensor([self.y[index]])
        return x,y
    
    def __len__(self):
        return self.length
    

#사용자 정의 모델
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2,1)

    def forward(self,x):
        x = self.layer(x)
        return x
    

#사용자 정의 데이터세트 및 데이터로더
dataset = CustomDataset('../datasets/non_linear.csv')
dataset_size = len(dataset)
train_size = int(dataset_size *0.8)
validation_size = int(dataset_size *0.1)
test_size = dataset_size - train_size - validation_size

train_dataset , validation_dataset, test_dataset = random_split(dataset, [train_size,validation_size,test_size])
print(f"Train data size : {len(train_dataset)}")
print(f"Validataion data size : {len(validation_dataset)}")
print(f"Test data size : {len(test_dataset)}")

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CustomModel().to(device)
criterion = nn.MSELoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(10000):
    cost = 0.0

    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss

    cost = cost / len(train_dataloader)

    if (epoch +1) % 1000 == 0:
        print(f"epoch : {epoch +1:4d}, Model: {list(model.parameters())}, Cost: {cost:.3f}")


with torch.no_grad():
    model.eval() # 평가 모드로 반환(특정 계층에서는 평가 모드와 학습 모드가 다름)
    for x, y in validation_dataloader:
        outputs = model(x)
        print(f"X : {x}")
        print(f"Y : {y}")
        print(f"Predict : {outputs}")
        print("=====================")