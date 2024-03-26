1. gpu compute capability 확인
(3.5 이상을 사용하고 있어야 함)
- https://developer.nvidia.com/cuda-gpus

2. gpu 드라이버 최신 버전으로 업데이트
- https://www.nvidia.com/Download/index.aspx

3. 파이토치 gpu에서 지원하는 cuda 툴킷 버전 확인
- https://pytorch.org/get-started/locally/
(파이토치 버전에 따른 cuda 버전 확인)

4. 버전에 맞는 cuda 툴킷 설치
- https://developer.nvidia.com/cuda-toolkit-archive

5. cuDNN(cuda 심층 신경망 라이브러리 설치)
- cuda 버전과 호환되는 압축 파일을 다운로드해 nvidia computing toolkit이 설치된 경로로 파일을 덮어씌움
- https://developer.nvidia.com/rdp/cudnn-archive

6. 환경 변수에 경로를 등록(윈도우/리눅스)

7. 패키지 매니저나 아나콘다로 GPU를 사용하는 파이토치 설치
- gpu cuda 11.8인 경우 
`$ pip install torch torchvision torchaudio --index-url`
    https://download.pytorch.org/whl/cu118

 
8. 파이토치를 설치했다면 설치된 버전과 gpu가속 확인
    
    ```{python}
    import torch

    print(torch.__version__)
    print(torch.cuda.is_available())
    ```
    