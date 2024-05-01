#데이터가 가진 고유한 특징을 유지한 채 변형하거나 노이즈를 추가해 데이터세트의 크기를 인위적으로 늘리는 방법
#학습 데이터 수집이 어려울 경우, 기존 학습 데이터를 재가공해 새로운 데이터 생성
# 과대적합을 줄이고 일반화 능력 향상 , 기존 데이터의 형질이 유지되므로 모델의 분산과 편향을 줄일 수 있음 
# 클래스 간 불균형 완화
# 노이즈 과ㅏ다 시 특징 파괴, 데이터 수집보다 더 많은 비용이 들 수도...

"""
텍스트 데이터
"""
# nlpaug 라이브러리를 활용해 텍스트 데이터를 증강

#%%
#삽입 및 삭제 
import nlpaug.augmenter.word as naw

texts = [
    "Those who can imagins anything, can create the impossible",
    "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
    "If a machine is expected to be infallible, it cannot also be intelligent."
]

aug = naw.ContextualWordEmbsAug(model_path="bert-base-uncased",action="insert")
#bert 모델을 활용해 단어를 삽입하는 기능 제공 (삽입, 교체 기능 제공)
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts,augmented_texts):
    print(f"src:{text}")
    print(f"dst:{augmented}")
    print("=======================")

# src:Those who can imagins anything, can create the impossible
# dst:and those who can imagins anything, thus can now create the beautiful impossible
# =======================
# src:We can only see a short distance ahead, but we can see plenty there that needs to be done.
# dst:we just can now only see a short distance out ahead, but then we can see plenty there anyway that needs hours to sometimes be done.
# =======================
# src:If a machine is expected to be infallible, it cannot also be intelligent.
# dst:if not a machine is falsely expected to be infallible, such it therefore cannot therefore also be automatically intelligent.
# =======================
#%%
#문자 삭제
import nlpaug.augmenter.char as nac

texts = [
    "Those who can imagins anything, can create the impossible",
    "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
    "If a machine is expected to be infallible, it cannot also be intelligent."
]

aug = nac.RandomCharAug(action="delete")
#무작위로 문자 삭제 (삽입,대체, 교체, 삭제 기능 제공)
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts,augmented_texts):
    print(f"src:{text}")
    print(f"dst:{augmented}")
    print("=======================")
# src:Those who can imagins anything, can create the impossible
# dst:Tos who can agin anything, can crat the impossible
# =======================
# src:We can only see a short distance ahead, but we can see plenty there that needs to be done.
# dst:We can nl see a shr dtane ahead, but we can see plenty thr tt eds to be ne.
# =======================
# src:If a machine is expected to be infallible, it cannot also be intelligent.
# dst:If a mine is epecd to be infallible, it anot ao be intelgn.
# =======================
#%%
#교체 및 대체
#교체(swap): 단어나 문자의 위치를 교환
#대체(substitute): 유사한 의미로 변경

#단어 대체(1)
import nlpaug.augmenter.word as naw

texts = [
    "Those who can imagins anything, can create the impossible",
    "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
    "If a machine is expected to be infallible, it cannot also be intelligent."
]

aug = naw.SynonymAug(aug_src="wordnet")
#wordnet db를 사용해 단어를 대체하여 데이터를 증강
#해당 기능은 문맥을 파악하는 것이 아니라, db내 유의어로 변경하므로 본래의 문맥과 전혀 다른 문장이 생성될 수 있음
#모델 대체 시 ContextualWordEmbsAug
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts,augmented_texts):
    print(f"src:{text}")
    print(f"dst:{augmented}")
    print("=======================")
# src:Those who can imagins anything, can create the impossible
# dst:Those who tin imagins anything, ass create the inconceivable
# =======================
# src:We can only see a short distance ahead, but we can see plenty there that needs to be done.
# dst:We can solely see a short distance ahead, but we potty see enough there that call for to cost done.
# =======================
# src:If a machine is expected to be infallible, it cannot also be intelligent.
# dst:If a machine constitute expected to be infallible, it cannot also follow intelligent.
# =======================-en-de
#%%
#단어대체 (2)
import nlpaug.augmenter.word as naw

texts = [
    "Those who can imagins anything, can create the impossible",
    "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
    "If a machine is expected to be infallible, it cannot also be intelligent."
]

reserved_tokens = [
    ["can","can't","cannot","could"],
]

reserved_aug = naw.ReservedAug(reserved_tokens=reserved_tokens)
#입력 데이터에 포함된 단어를 특정한 단어로 대체

augmented_texts = reserved_aug.augment(texts)

for text, augmented in zip(texts,augmented_texts):
    print(f"src:{text}")
    print(f"dst:{augmented}")
    print("=======================")
#%%
#역변역
#입력 텍스트를 특정 언어로 번역한 다음 다시 본래의 언어로 번역
# 증강 방법 중 가장 많은 리소스 소모
import nlpaug.augmenter.word as naw

texts = [
    "Those who can imagins anything, can create the impossible",
    "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
    "If a machine is expected to be infallible, it cannot also be intelligent."
]

back_translation = naw.BackTranslationAug(
                    from_model_name='facebook/wmt19-en-de', #입력 모델: 영어->독일어
                    to_model_name='facebook/wmt19-de-en' # 출력 모델: 독일어->영어
)
#입력 데이터에 포함된 단어를 특정한 단어로 대체

augmented_texts = back_translation.augment(texts)

for text, augmented in zip(texts,augmented_texts):
    print(f"src:{text}")
    print(f"dst:{augmented}")
    print("=======================")