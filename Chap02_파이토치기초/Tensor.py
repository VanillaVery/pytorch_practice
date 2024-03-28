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

"""장치 변환"""
cpu = torch.FloatTensor([1,2,3])
gpu = cpu.cuda()
gpu2cpu = gpu.cpu()
cpu2gpu = cpu.to("cuda")

"""넘파이 배열의 텐서 변환"""
import numpy as np

ndarray = np.array([1,2,3],dtype = np.uint8)
print(torch.tensor(ndarray))
print(torch.Tensor(ndarray))
print(torch.from_numpy(ndarray))

"""텐서의 넘파이 배열 변환"""
#추론된 결과를 후처리하거나 결과값 활용 시

tensor = torch.cuda.FloatTensor([1,2,3])
ndarray = tensor.detech().cpu().numpy()

