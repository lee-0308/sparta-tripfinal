import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from matplotlib import cm
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


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "Files"
IMG_DIR = BASE_DIR / "Images"

# ------------------------
# 1. 데이터 불러오기
# ------------------------
df = pd.read_csv(DATA_DIR / "가성비중심_분석용.csv")
df.rename(columns={'cluster': 'Cluster'}, inplace=True)

# ------------------------
# 2. 기본 설정
# ------------------------
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# ------------------------
# 3. 사이드바
# ------------------------
st.sidebar.header("필터 조건 설정")
selected_clusters = st.sidebar.multiselect("클러스터 선택", sorted(df['Cluster'].unique()), default=sorted(df['Cluster'].unique()))
filtered_df = df[df['Cluster'].isin(selected_clusters)]

city_options = ["모두 보기"] + sorted(filtered_df['City'].unique())
selected_city = st.sidebar.selectbox("도시 선택", city_options)
if selected_city != "모두 보기":
    filtered_df = filtered_df[filtered_df['City'] == selected_city]

min_cost = float(df['Estimated_Total_Cost'].min())
max_cost = float(df['Estimated_Total_Cost'].max())
selected_range = st.sidebar.slider("총 경비 범위 ($)", min_value=min_cost, max_value=max_cost, value=(min_cost, max_cost))
filtered_df = filtered_df[
    (filtered_df['Estimated_Total_Cost'] >= selected_range[0]) & 
    (filtered_df['Estimated_Total_Cost'] <= selected_range[1])
]

# ------------------------
# 4. 제목 및 요약
# ------------------------
st.title("가성비 중심 도시 군집 분석")

col_exp, col_radar = st.columns([1.2, 1.2])

with col_exp:
    for _ in range(5):  # 여백 3줄 생성
        st.write("")

    st.markdown("<h4 style='font-weight:bold;'> 분석에 사용된 주요 변수:</h4>", unsafe_allow_html=True)
    st.markdown("""
    - **Estimated_Total_Cost (총 여행 경비)**  
    - **Avg_Hotel_Price_USD (1박 호텔 가격)**  
    - **Daily_Food_Cost (하루 식비)**  
    - **Daily_Transit_Cost (하루 교통비)**  
    """)

with col_radar:
    st.subheader(" ")

    def plot_radar_chart_z(df_means):
        scaler = StandardScaler()
        df_z = pd.DataFrame(scaler.fit_transform(df_means), columns=df_means.columns)

        categories = df_z.columns.tolist()
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig = plt.figure(figsize=(5.5, 4.5))
        cmap = cm.get_cmap('Set2', df_z.shape[0])

        for i, row in df_z.iterrows():
            values = row.tolist() + [row.tolist()[0]]
            plt.polar(angles, values, label=f'Cluster {i}',
                      linewidth=2.0, marker='o', markersize=5, color=cmap(i))
            plt.fill(angles, values, alpha=0.15, color=cmap(i))

        plt.xticks(angles[:-1], categories, fontsize=11)
        plt.yticks([-2, -1, 0, 1, 2], ['-2', '-1', '0', '1', '2'], color='gray', fontsize=9)
        plt.title(" ", fontsize=13)
        plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05))
        plt.tight_layout()
        return fig

    # 클러스터 필터링 반영
    cluster_means = filtered_df.groupby('Cluster')[
        ['Estimated_Total_Cost', 'Avg_Hotel_Price_USD', 'Daily_Food_Cost', 'Daily_Transit_Cost']
    ].mean()

    if not cluster_means.empty:
        st.pyplot(plot_radar_chart_z(cluster_means))
    else:
        st.warning("선택한 클러스터에 해당하는 데이터가 없습니다.")


# ------------------------
# 5. 군집별 소비자 유형 (해석표)
# ------------------------
st.subheader(" 군집별 소비자 유형 ")
st.markdown("""
<style>
.custom-table {
    border-collapse: collapse;
    width: 100%;
    font-size: 15px;
}
.custom-table th, .custom-table td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: center;
}
.custom-table th {
    background-color: #f2f2f2;
    color: black;
    font-weight: bold;
}
.custom-table tr:nth-child(even) {background-color: #f9f9f9;}
.custom-table tr:hover {background-color: #f1f1f1;}
</style>

<table class="custom-table">
    <tr>
        <th>클러스터</th>
        <th>도시 수</th>
        <th>총비용(USD)</th>
        <th>특징</th>
        <th>소비자 유형</th>
    </tr>
    <tr>
        <td>Cluster 0</td>
        <td>24개</td>
        <td>124</td>
        <td>교통비 비중 높음, 전반적으로 균형 있음</td>
        <td><b style="color:#66c2a5">가성비형</b></td>
    </tr>
    <tr>
        <td>Cluster 1</td>
        <td>16개</td>
        <td>202</td>
        <td>숙소비, 특히 식비가 매우 높음</td>
        <td><b style="color:#8da0cb">미식가형</b></td>
    </tr>
    <tr>
        <td>Cluster 2</td>
        <td>15개</td>
        <td>184</td>
        <td>숙소비/식비 높지만 Cluster 1보단 저렴</td>
        <td><b style="color:#ffd92f">So-so형</b></td>
    </tr>
    <tr>
        <td>Cluster 3</td>
        <td>12개</td>
        <td>290</td>
        <td>숙소비가 압도적, 전체 비용 최고</td>
        <td><b style="color:#bdbdbd">초고가형</b></td>
    </tr>
</table>
""", unsafe_allow_html=True)


col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("총 도시 수", f"{len(df['City'].unique()):,}개")
with col2:
    st.metric("평균 총 경비", f"${df['Estimated_Total_Cost'].mean():,.0f}")
with col3:
    st.metric("평균 호텔비", f"${df['Avg_Hotel_Price_USD'].mean():,.0f}")
with col4:
    st.metric("평균 식비", f"${df['Daily_Food_Cost'].mean():,.0f}")
with col5:
    st.metric("평균 교통비", f"${df['Daily_Transit_Cost'].mean():,.0f}")
st.markdown("<br><br>", unsafe_allow_html=True)

#--------------------------
# 6. 클러스터별 평균경비
# ------------------------
import seaborn as sns
import matplotlib.pyplot as plt

st.subheader("클러스터별 변수 평균 비교 (Bar Chart)")

# 1. 사용할 변수 리스트
bar_cols = ['Estimated_Total_Cost', 'Avg_Hotel_Price_USD', 'Daily_Food_Cost', 'Daily_Transit_Cost']

# 2. 선택된 클러스터 기준으로 평균 집계
bar_means = filtered_df.groupby("Cluster")[bar_cols].mean().reset_index()

# 3. 시각화를 위한 melt
bar_melted = bar_means.melt(id_vars='Cluster', var_name='변수', value_name='값')

# 🔹 클러스터 색상 고정
cluster_color_map = {
    0: '#66c2a5',  # 청록
    1: '#8da0cb',  # 연보라
    2: '#ffd92f',  # 노랑
    3: '#bdbdbd'   # 회색
}

# 🔹 hue 순서 맞춰 색상 리스트 추출
unique_clusters = sorted(bar_melted['Cluster'].unique())
palette = [cluster_color_map[c] for c in unique_clusters]

# 4. 그래프 생성
fig_bar, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=bar_melted, x='변수', y='값', hue='Cluster', palette=palette, ax=ax)

# 5. 막대 위에 값 표시 (한 자리 소수로 포맷팅)
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', label_type='edge', padding=3)

# 6. 그래프 설정

ax.set_ylabel("평균 비용 (USD 기준)")
ax.set_xlabel("지표")
ax.legend(title='Cluster', loc='upper right')
plt.xticks(rotation=15)
plt.tight_layout()

# 7. 출력
st.pyplot(fig_bar)




# ------------------------
## 시각화 2 클러스터별 주요 지표 히트맵
# ------------------------
import seaborn as sns
import matplotlib.pyplot as plt

st.subheader("클러스터별 주요 지표 히트맵")

# 1. 사용할 변수
heatmap_cols = ['Estimated_Total_Cost', 'Avg_Hotel_Price_USD', 'Daily_Food_Cost', 'Daily_Transit_Cost']

# 2. 클러스터별 평균값 계산
cluster_means_for_heatmap = df.groupby('Cluster')[heatmap_cols].mean().round(1)

# 3. 시각화
fig_hm, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(cluster_means_for_heatmap, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax)

# 4. 그래프 설정

ax.set_xlabel("지표")
ax.set_ylabel("클러스터")
plt.tight_layout()

# 5. 출력
st.pyplot(fig_hm)


# ------------------------
# 6. 군집별 평균 테이블
# ------------------------
st.subheader(" 군집별 평균 통계")
st.dataframe(cluster_means.round(2).reset_index())
# ------------------------
# 7. 필터링 결과 테이블
# ------------------------
st.subheader(" 필터링된 도시 정보")
st.dataframe(filtered_df)

st.markdown("<br><br>", unsafe_allow_html=True)
# ------------------------
# 8. 도시 위치 시각화
# ------------------------
st.subheader(" 도시 위치 시각화 (국가 기준 지도)")
# Cluster를 문자열로 변환해서 색 구분 명확히
filtered_df['Cluster'] = filtered_df['Cluster'].astype(str)

# 클러스터별 고정 색상 정의
color_map = {
    '0': '#66c2a5',  
    '1': '#8da0cb',  
    '2': '#ffd92f',  
    '3': '#bdbdbd'   
}

fig2 = px.choropleth(
    filtered_df,
    locations='country',
    locationmode='country names',
    color='Cluster',
    hover_name='City',
    color_discrete_map=color_map
)
st.plotly_chart(fig2, use_container_width=True)


