import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import platform
from pathlib import Path
from matplotlib import font_manager, rc
import os

# 나눔고딕 경로 (Streamlit Cloud 기준)
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# 폰트 존재하면 설정
if os.path.exists(font_path):
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지
else:
    print("⚠️ 서버에 한글 폰트가 없습니다.")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Files"
IMG_DIR = BASE_DIR / "Images"


# ------------------------
# 1. 데이터 불러오기
# ------------------------
df = pd.read_csv(DATA_DIR / "Climate_Cluster1.csv")
df2 = pd.read_csv(DATA_DIR / "Countries_with_cluster.csv")

# ------------------------
# 2. 제목 & 설명
# ------------------------
st.title("기후 중심 45개국 군집 분석")
st.header("1. 기초 정보")

st.markdown("""
**군집 분석에 사용한 기후 데이터**

1. 각 국가의 1월 평균 기온  
2. 각 국가의 7월 평균 기온  
3. 연교차 *(7월 기온 - 1월 기온)*  
4. 한국과의 연교차 차이 *(한국보다 연교차가 크면 양수, 작으면 음수)*
""")
st.subheader("군집별 통계")
st.dataframe(df)



# ------------------------
# 3. 요약 통계
# ------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("총 국가 수", f"{len(df2):,}곳")
with col2:
    st.metric("1월 평균기온", f"{df['temp_01'].mean():.1f}℃")
with col3:
    st.metric("7월 평균기온", f"{df['temp_07'].mean():.1f}℃")
with col4:
    st.metric("평균 연교차", f"{df['annual_temp_range'].mean():.1f}℃")

# ------------------------
# 4. 필터 선택
# ------------------------
st.header("2. 클러스터 및 국가 필터")

# 클러스터 선택
selected_clusters = st.multiselect("클러스터 선택", options=sorted(df2['Cluster'].unique()), default=sorted(df2['Cluster'].unique()))

# 필터링
filtered_df = df2[df2['Cluster'].isin(selected_clusters)]

# 국가 선택
country_options = ["모두 보기"] + sorted(filtered_df['country'].unique())
selected_country = st.selectbox("국가 선택", options=country_options)

# 국가 필터 적용
if selected_country != "모두 보기":
    filtered_df = filtered_df[filtered_df['country'] == selected_country]

# 연교차 범위 슬라이더
min_temp = float(df2['annual_temp_range'].min())
max_temp = float(df2['annual_temp_range'].max())
selected_range = st.slider("연교차 범위 선택 (℃)", min_value=min_temp, max_value=max_temp, value=(min_temp, max_temp))

# 연교차 범위 필터
filtered_df = filtered_df[(filtered_df['annual_temp_range'] >= selected_range[0]) & (filtered_df['annual_temp_range'] <= selected_range[1])]

# ------------------------
# 5. 데이터프레임 출력
# ------------------------
st.subheader("📋 필터링된 국가 정보")
st.dataframe(filtered_df)

# ------------------------
# 6. 시각화 (matplotlib)
# ------------------------


st.subheader("3. 클러스터별 기온 비교 (Matplotlib)")

if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

x = df['Cluster']
y1 = df['temp_01']
y2 = df['annual_temp_range']

fig, ax1 = plt.subplots(figsize=(10, 6))

# 1월 평균 기온 (선)
line1, = ax1.plot(x, y1, color='blue', marker='o', label='1월 평균 기온 (℃)')
ax1.set_xlabel("군집(클러스터)")
ax1.set_ylabel("1월 평균 기온 (℃)", color='blue')

# 연교차 (막대)
ax2 = ax1.twinx()
bars = ax2.bar(x, y2, alpha=0.5, color='red')
ax2.set_ylabel("연교차 (℃)", color='red')

# ✅ 범례 수동 지정 (Patch를 직접 만듦)
line_legend = line1
bar_legend = mpatches.Patch(color='red', alpha=0.5, label='연교차 (℃)')

ax1.legend(handles=[line_legend, bar_legend], loc='upper left')

plt.title('세계 도시들의 1월 평균 기온과 연교차에 대한 군집')

fig.tight_layout()
st.pyplot(fig)


#---------------------------------------
# 데이터 생성
x = df['Cluster']
y1 = df['temp_07']
y2 = df['temp_diff_korea']

# 그래프 생성 및 이중 축 설정
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# 각 축에 그래프 그리기
line1, = ax1.plot(x, y1, 'k-', marker='o', label='7월 평균 기온(℃)') # y1 그래프 녹색 실선
line2, = ax2.plot(x, y2, 'b-', marker='o', label='한국과 비교한 연교차 차이 (℃)') # y2 그래프 파란색 실선

# 축 레이블 설정
ax1.set_xlabel('군집(클러스터)')
ax1.set_ylabel('7월 평균 기온 (℃)')
ax2.set_ylabel('한국과의 연교차 차이 정도 (℃)', color='b')

# 범례 표시 (위치 조정)
lines = [line1, line2]
labels = [l.get_label() for l in lines]
plt.legend(lines, labels, loc='lower left') # 범례 위치 조정

# x축 눈금 간격 설정 (0부터 10까지 2 간격으로)
plt.xticks(np.arange(0, 5, 1))
plt.title('7월 평균 기온과, 한국과의 연교차 비교값에 대한 군집')

fig.tight_layout()
st.pyplot(fig)

#----------------------------------------

# 데이터 생성
x = df['Cluster']
y1 = df['annual_temp_range']
y2 = df['creditcard_ratio']

# 그래프 생성 및 이중 축 설정
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# 각 축에 그래프 그리기
line1, = ax1.plot(x, y1, 'm-', marker='o', label='연교차 (℃)') # y1 그래프 녹색 실선
line2, = ax2.plot(x, y2, 'g-', marker='o', label='신용카드 사용률 (%)') # y2 그래프 파란색 실선

# 축 레이블 설정
ax1.set_xlabel('군집(클러스터)')
ax1.set_ylabel('연교차 (℃)', color='m')
ax2.set_ylabel('신용카드 사용률 (%)', color='g')

# x축 눈금 간격 설정 (0부터 10까지 2 간격으로)
plt.xticks(np.arange(0, 5, 1))

# 범례 표시 (위치 조정)
lines = [line1, line2]
labels = [l.get_label() for l in lines]
plt.legend(lines, labels, loc='lower left') # 범례 위치 조정
plt.title('연교차와 신용카드 사용률에 대한 군집')

fig.tight_layout()
st.pyplot(fig)

st.markdown("3번째 그래프는 의미심장하다. **신용카드 사용률이 높은 국가는 중진국 이상~선진국 수준이며 선진국일수록 카드 사용률이 높게 나타난다.** 이는 여러 요인이 있으나, 금융 체계와 사회 안전망이 갖춰져야 비로소 신용 카드를 쓸 수 있는 사회가 되기 때문이라고 한다. **그런데 연교차가 큰 중위도 국가들이 대체로 선진국**이라서 **둘의 관계는 정비례 관계**를 띠는 경향을 보이고 있다.")


# ------------------------
# 7. Plotly 산점도
# ------------------------
st.subheader("4. 클러스터별 국가 분포 (산점도)")

fig1 = px.scatter(
    filtered_df,
    x='Latitude',
    y='Longitude',
    color='Cluster',
    size='temp_07',
    hover_data=['country', 'annual_temp_range', 'temp_01', 'temp_07', 'creditcard_ratio', 'Religion', 'Travel Safety Score'],
    labels={'Latitude': '위도', 'Longitude': '경도'},
    title='국가 분포 (위도 vs 경도)',
    color_continuous_scale='Viridis'
)
st.plotly_chart(fig1, use_container_width=True)

# ------------------------
# 8. Plotly 지도
# ------------------------
st.subheader("5. 세계 지도 시각화")

fig2 = px.scatter_geo(
    filtered_df,
    lat='Latitude',
    lon='Longitude',
    color='Cluster',
    size='temp_07',
    hover_name='country',
    projection="natural earth",
    title="클러스터별 국가 분포 (지도 기반)",
    color_continuous_scale='Viridis'
)
st.plotly_chart(fig2, use_container_width=True)




    















    








