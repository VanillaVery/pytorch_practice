# 회전, 대칭, 이동, 크기 조정 등
# torchvision imgaug 라이브러리

#%%
# 통합 클래스 및 변환 적용 방식
from PIL import Image
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize(size=(512,512)), # 이미지 데이터를 512,512크기로 변환
        transforms.ToTensor() #텐서 타입으로 변환(PIL.image -> Tensor) / minmax정규화 / 채널, 높이, 너비 형태로 변경
    ]
)

image = Image.open("../datasets/images/cat.jpg")
transformed_image = transform(image)

print(transformed_image.shape)
#%%
#회전 및 대칭
