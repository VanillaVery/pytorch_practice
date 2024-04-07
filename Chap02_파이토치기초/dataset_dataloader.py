#데이터세트; 데이터의 집합 

#데이터는 파일 경로로 제공되거나, 전처리 단계가 필요한 경우도 있음
#데이터를 변형하고 매핑하는 코드를 학습과정에 직접 반영하면 
# 모듈화, 재사용성, 가독성을 떨어뜨리는 주요 원인이 됨

#이러한 현상을 방지, 코드를 구조적으로 설계할 수 있도록 
#데이터세트와 데이터로더를 사용 

#%%
#데이터세트
#학습에 필요한 데이터 샘플을 정제하고 정답을 저장하는 기능 제공
#데이터 세트 클래스의 기본형 
class Dataset:
    def __init__(self,data,*arg,**kwargs):
        self.data=data
    
    def __getitem__(self,index):
        return tuple(data[index] for data in data.tensors)
    
    def __len__(self):
        return self.data[0].size(0)