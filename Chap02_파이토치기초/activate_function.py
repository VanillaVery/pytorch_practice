import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
"""
이진 분류: 파이토치
"""
# 사용자 정의 데이터세트 선언 방법
class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        self.x1 = df.iloc[:,0].values
        self.x2 = df.iloc[:,1].values
        self.x3 = df.iloc[:,2].values
        self.y = df.iloc[:,3].values
        self.length = len(df)

    def __getitem__(self, index): #호출 메서드
        x = torch.FloatTensor([self.x1[index], self.x2[index], self.x3[index]])
        y = torch.FloatTensor([self.y[index]])
        return x, y
    
    def __len__(self):
        return self.length
    
#사용자 정의 모델 선언
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__() #Module 클래스의 속성을 초기화
        self.layer = nn.Sequential( # 여러 계층을 하나로 묶음. 가독성 up
            nn.Linear(3,1),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        x = self.layer(x)
        return x

#이진 교차 엔트로피
criterion = nn.BCELoss().to(device)
