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

#%%
"""
모델 상태 저장/불러오기
"""
#모델의 매개변수만 저장하여 활용 
#state_dict()메서드로 모델 상태를 저장 -> orderdict 형식으로 반환됨
import torch
from torch import _nnpack_available

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2,1)

    def forward(self,x):
        x = self.layer(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"    
model = CustomModel().to(device)

model_state_dict = torch.load("../models/model_state_dict.pt",map_location=device)
model.load_state_dict(model_state_dict) #CustomModel에 학습 결과를 반영
#모델 상태만 불러오면 모델 구조를 알 수 없으므로 클래스가 동일하게 구현되어 있어야 함

#%%
"""
체크포인트 저장/불러오기
"""
#체크포인트 저장
import torch
import pandas as pd
from torch import nn 
from torch import optim
from torch.utils.data import Dataset, DataLoader


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

checkpoint = 1
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
        torch.save(
            {
                "model":"CustomModel",
                "epoch":epoch,
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "cost":cost,
                "description":f"CustomModel 체크포인트-{checkpoint}"
            },
            f"../models/checkpoint-{checkpoint}.pt"
        )
        # 다양한 정보를 저장
        # 학습을 이어 진행해야 하므로 에폭, 모델 상태, 최적화 상태는 필수로 포함
        checkpoint += 1
#%%
#체크포인트 불러오기
import torch
import pandas as pd
from torch import nn 
from torch import optim
from torch.utils.data import Dataset, DataLoader

# 중략...

checkpoint = torch.load("../models/checkpoint-6.pt")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
checkpoint_epoch = checkpoint["epoch"]
checkpoint_description = checkpoint["description"]
print(checkpoint_description)

for epoch in range(checkpoint_epoch+1,10000):
    cost = 0.0

    for x,y in train_dataloader:
        x = x.to(device)
        y = y.to(device)

        output = model(x)
        loss = criterion(output,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch : {epoch +1:4d}, Model : {list(model.parameters())}, Cost: {cost:.3f}")