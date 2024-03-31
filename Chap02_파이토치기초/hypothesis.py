# 가설: 독립변수와 y간의 관계를 가장 잘 근사하기 위해 사용

# paired t-test: 동일한 항목 또는 그룹을 두 번 테스트할 때 사용 
# unpaired t-test: 등분산성을 만족하는 두 개의 독립적 그룹 간의 평균을 비교 

#성별에 따른 키 차이 시각화
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt

man_height = stats.norm.rvs(loc=170, scale=10,size=500,random_state=1)
woman_height = stats.norm.rvs(loc=150, scale=10,size=500,random_state=1)

X = np.concatenate([man_height,woman_height])
y = ["man"] * len(man_height) + ['woman'] * len(woman_height)

df = pd.DataFrame(list(zip(X,y)),columns=['X','y'])
fig = sns.displot(data=df, x="X",hue="y",kind="kde")
fig.set_axis_labels("cm","count")
plt.show()

#%%
#unpaired t-test
