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

