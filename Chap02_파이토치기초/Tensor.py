import torch

"""텐서 생성"""
torch.tensor([1,2,3]) # -> int 
# 입력된 데이터를 복사해 텐서로 변환 /자동으로 자료형 할당
torch.Tensor([[1,2,3],[4,5,6]]) #-> float
# 텐서의 기본형으로 텐서 인스턴스를 생성 / 
# 자료형이 명확히 표현되므로 권장됨

torch.LongTensor([1,2,3])
torch.FloatTensor([1,2,3]) 

"""텐서 속성"""
tensor = torch.rand(1,2)
print(tensor)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)

"""차원 변환"""
tensor = torch.rand(1,2)
print(tensor)
print(tensor.shape)

tensor = tensor.reshape(2,1)
print(tensor)
print(tensor.shape)

"""자료형 설정"""
tensor =  torch.rand((3,3),dtype=torch.float)
print(tensor)
#float와 torch.float는 다름 
#텐서 선언시 가능한 한 torch.* 형태로 명확히 선언

"""장치 설정"""
device = "cuda" if torch.cuda.is_available() else "cpu"
cpu = torch.FloatTensor([1,2,3])
gpu = torch.cuda.FloatTensor([1,2,3])
tensor = torch.rand((1,1),device=device)
print(device)
print(cpu)
print(gpu)
print(tensor)
