import pandas as pd
import numpy as np
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

# 정규화 시키기
from sklearn.preprocessing import StandardScaler
data_consumption = pd.DataFrame(StandardScaler().fit_transform(data.iloc[:, 7:16]), columns=data.columns[7:16])
data_consumption.info()  # 칼럼이름과 결측값, 데이터 형태 표시

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
cluster_center = pd.DataFrame(kmeans_model.cluster_centers_, columns=data.columns[7:16])

from collections import Counter

Counter(kmeans_model.predict(data_consumption))
data_consumption['pred'] = kmeans_model.predict(data_consumption)