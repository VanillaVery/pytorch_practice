#모델이 작고 변동성이 거의 없는 경우, 간단하게 가중치를 초기화 할 수 있음
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(1,2),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(2,1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.layer[0].weight)
        self.layer[0].bias.data.fill_(0.01)

        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.01)
#%%
# 모델의 코드가 복잡할 때, 가중치 초기화 메서드를 모듈화 해 적용 
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(1,2),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(2,1)
        self.apply(self._init_weights) 
        #가중치 초기화 메서드를 범용적으로 변경 시, torch.apply를 사용 
        # 적용 함수는 텐서의 각 요소에 임의의 함수를 적용하고 결과와 함께 새 텐서를 반환

    def _init_weights(self,module):
        # module 매개변수: 모델의 매개변수
        if isinstance(module,nn.Linear):# module이 선형 변환 함수인지 확인
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.01)
        print(f"Apply : {module}")

model = Net()

