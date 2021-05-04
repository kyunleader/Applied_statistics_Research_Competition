import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import matplotlib

font_name = font_manager.FontProperties(fname='C:/Windows/Fonts/H2HDRM.TTF').get_name()
plt.rc('font', family=font_name)
matplotlib.rcParams['axes.unicode_minus'] = False  # matplotlib 에 한국어 폰트 적용

# 데이터 불러오기
data = pd.read_excel('data.xlsx')

# 데이터 탐색
''' 간단하게 해보기 (jupyter 사용)
import sweetviz as sv
advert_report = sv.analyze(data)
sv.analyze(data).show_html('./sweetviz_Advertising.html')'''

# 남여 비교
sns.set_style('whitegrid')
colors = ['#FF1E00', '#4380D3']  # 색 설정
sns.set_palette(colors)
plt.rc('font', family=font_name)  # 폰트 설정
sns.countplot(x=['남' if i == 1 else '여' for i in data['sex']])

# 학년 비교
sns.countplot(x='grade', data=data, palette='Accent')
plt.xlabel('학년')

# 연애 유무
sns.countplot(x=['커플' if i == 1 else '솔로' for i in data['couple']], palette='Accent')

# 한달 지출비 분포
plt.figure(figsize=(9, 9))
for i in range(1, 10):
    print(i)
    plt.subplot(3, 3, i)
    sns.distplot(data.iloc[:, i + 6])

# k means 군집분석 해보기
from sklearn.cluster import KMeans

# 군집분석 데이터 준비
data_consumption = data.iloc[:, 46:55]

'''kmeans_model = KMeans(n_clusters=3, random_state=486)
kmeans_model.fit(data_consumption)
pred = kmeans_model.predict(data_consumption)
list(pred).index(2)'''


# elbow 포인트 구하기
def elbow(x):
    sse = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=486)
        km.fit(x)
        sse.append(km.inertia_)

    plt.plot(range(1, 11), sse, marker='o')
    plt.xlabel('클러스터 개수')
    plt.ylabel('sse')
    plt.show()


elbow(data_consumption)  # 엘보우 포인트가 크게 보이지 않으나 보고서 내용과 일치시키기 위해 3개 설정

# 클러스터의 중심점 보기
kmeans_model = KMeans(n_clusters=3, random_state=486)
kmeans_model.fit(data_consumption)
pd.DataFrame(kmeans_model.cluster_centers_, columns=data.columns[7:16])

# 클러스터 별 주요 차이를 보이는 식비, 패션/쇼핑, 저축의 변수만을 가지고 다시 클러스터링
data_main = data_consumption[['eatratio', 'fashion/shoppingratio', 'savingratio']]

elbow(data_main)  # elbow point 확인

kmeans_model = KMeans(n_clusters=3, random_state=486)
kmeans_model.fit(data_main)
cluster_center = pd.DataFrame(kmeans_model.cluster_centers_, columns=data_main.columns)

Counter(kmeans_model.predict(data_main))  # 집단의 편향성이 있는지 확인

data_main['pred'] = kmeans_model.predict(data_main)

# 데이터 3d plot 그려보기

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data_main['eatratio'], data_main['fashion/shoppingratio'], data_main['savingratio'], c=data_main['pred'])
# 어느정도 나뉘어 진 것을 확인

# 한달 지출비가 2배가 되었을 때 클러스터 확인
# 식비, 패션/쇼핑, 저축 변수를 활용하여 클러스터 분석

data_main_x2 = data[['eatratio2', 'fashion/shoppingratio2', 'savingratio2']]

kmeans_model = KMeans(n_clusters=3, random_state=486)
kmeans_model.fit(data_main_x2)
cluster_center = pd.DataFrame(kmeans_model.cluster_centers_, columns=data_main_x2.columns)

Counter(kmeans_model.predict(data_main_x2))  # 집단의 편향성이 있는지 확인

data_main_x2['pred_x2'] = kmeans_model.predict(data_main_x2)

# 데이터 3d plot 그려보기

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(data_main_x2['eatratio2'], data_main_x2['fashion/shoppingratio2'],
           data_main_x2['savingratio2'], c=data_main_x2['pred_x2'])




# group by 나누어서 3d plot 표시 하기
data_main_x2.columns = ['eatratio2', 'shoppingratio2', 'savingratio2', 'pred_x2']  # '/'이게 안되서 지움
groups = data_main_x2.groupby('pred_x2')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for name, group in groups:
    print(name, group)
    ax.scatter(group.eatratio2, group.shoppingratio2,
           group.savingratio2, label=name)
ax.legend()
ax.set_xlabel('식비')
ax.set_ylabel('쇼핑')
ax.set_zlabel('저축')

'''
군집분석 결과로 봤을 때
pred, pred_x2 변수에서 각각 0,1,2는 식비형, 저축형, 쇼핑형이라고 볼 수 있다.
'''

# 변수 바꾸기
data_main['pred_x2'] = data_main_x2['pred_x2']
data_main.loc[data_main['pred'] == 0, 'pred'] = '식비형'
data_main.loc[data_main['pred'] == 1, 'pred'] = '저축형'
data_main.loc[data_main['pred'] == 2, 'pred'] = '쇼핑형'

data_main.loc[data_main['pred_x2'] == 0, 'pred_x2'] = '식비형'
data_main.loc[data_main['pred_x2'] == 1, 'pred_x2'] = '저축형'
data_main.loc[data_main['pred_x2'] == 2, 'pred_x2'] = '쇼핑형'


# 식비형에 분포 되었던 사람이 한 달 지출이 2배가 되었을 때 어느 곳으로 이동하는지 시각화

eat_ra = data_main[data_main['pred'] == '식비형'].groupby(['pred_x2'])['pred_x2'].count()
plt.figure()
plt.pie(eat_ra.values,
        labels= eat_ra.index,
        shadow= True,
        autopct= '%1.2f%%')

# 저축, 쇼핑형도 같은 방법으로 시각화

save_ra = data_main[data_main['pred'] == '저축형'].groupby(['pred_x2'])['pred_x2'].count()
explode = [0.05, 0.05, 0.05]
plt.figure()
plt.pie(save_ra.values,
        labels= save_ra.index,
        shadow= True,
        autopct= '%1.2f%%',
        explode=explode)  # 저축형

sh_ra = data_main[data_main['pred'] == '쇼핑형'].groupby(['pred_x2'])['pred_x2'].count()
wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 5}
plt.figure()
plt.pie(sh_ra.values,
        labels= sh_ra.index,
        autopct= '%1.2f%%',
        wedgeprops=wedgeprops)

# 한달 지출비가 2배가 되었을때 성별, 연애, 아르바이트 여부에 따라 차이가 있는지

data['pred'] = data_main['pred']
data['pred_x2'] = data_main['pred_x2']

pd.crosstab(data.pred_x2, data.sex)  # 빈도표 만들기


import scipy.stats as stats
stats.chi2_contingency(observed=pd.crosstab(data.pred_x2, data.sex))
# (1) 카이스퀘어 값, 검정 통계량 (2) p-value (3) 자유도 (4)각각의 기대빈도
# 성별에 따라 지출 유형에 차이가 있다고 할 수 있다.



