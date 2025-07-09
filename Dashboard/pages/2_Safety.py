import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import google.generativeai as genai
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "Files"
IMG_DIR = BASE_DIR / "Images"

# ------------------------
# 1. 데이터 불러오기 및 전처리
# ------------------------
df = pd.read_excel(DATA_DIR / "world_crime_dashborad.xlsx")
df.rename(columns={
    'cluster': 'Cluster',
    'cv(Homicide, 10만 명 당)': 'Homicide',
    'cv(Assault, 10만 명 당)': 'Assault',
    'cv(Kidnapping, 10만 명 당)': 'Kidnapping',
    'cv(SexualViolence, 10만 명 당)': 'Sexual Violence'
}, inplace=True)

# ------------------------
# 2. 기본 설정 (한글 폰트)
# ------------------------
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# ------------------------
# 3. 사이드바 필터
# ------------------------
st.sidebar.header("필터 조건 설정")
selected_clusters = st.sidebar.multiselect("클러스터 선택", sorted(df['Cluster'].unique()), default=sorted(df['Cluster'].unique()))
filtered_df = df[df['Cluster'].isin(selected_clusters)].copy()

country_options = ["모두 보기"] + sorted(filtered_df['Country'].unique())
selected_country = st.sidebar.selectbox("국가 선택", country_options)
if selected_country != "모두 보기":
    filtered_df = filtered_df[filtered_df['Country'] == selected_country]

# ------------------------
# 4. 페이지 제목 및 변수 소개
# ------------------------
st.title("🌍 세계 국가 범죄율 기반 군집 분석 대시보드")

st.markdown("#### 📌 분석에 사용된 주요 범죄 지표:")
st.markdown("- **살인 (Homicide)**\n- **폭행 (Assault)**\n- **아동납치 (Kidnapping)**\n- **성폭력 (Sexual Violence)**")

# ------------------------
# 5. Radar Chart
# ------------------------
st.subheader("📊 클러스터별 주요 범죄 지표 (Radar Chart)")

def plot_radar_chart_z(df_means):
    scaler = StandardScaler()
    df_z = pd.DataFrame(scaler.fit_transform(df_means), columns=df_means.columns)
    angles = np.linspace(0, 2*np.pi, len(df_z.columns), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 5))
    for i, row in df_z.iterrows():
        values = row.tolist() + [row.tolist()[0]]
        plt.polar(angles, values, label=f'Cluster {i}')
        plt.fill(angles, values, alpha=0.1)
    plt.xticks(angles[:-1], df_z.columns)
    plt.legend(loc='upper right')
    return fig

cluster_means = filtered_df.groupby('Cluster')[['Homicide', 'Assault', 'Kidnapping', 'Sexual Violence']].mean()
if not cluster_means.empty:
    st.pyplot(plot_radar_chart_z(cluster_means))
else:
    st.warning("선택한 클러스터에 해당하는 데이터가 없습니다.")

# ------------------------
# 6. 클러스터 명칭 요약표 (심플 & 모던)
# ------------------------
st.subheader("🧩 클러스터 명칭 요약표 – 심플 & 모던")

cluster_table = pd.DataFrame({
    "클러스터 번호": [0, 1, 2, 3],
    "명칭": ["🟢 SafeZone", "🟠 CityRisk", "🔵 LethalZone", "🟣 NoGoZone"],
    "설명": [
        "전반적으로 안전한 국가군",
        "폭행률이 높은 도시 중심 위험지대",
        "살인율이 높은 치명적 위험권",
        "살인 및 폭행 모두 높은 출국 비권장지"
    ]
})
st.table(cluster_table)

# ------------------------
# 7. 히트맵 (정규화된 지표)
# ------------------------
st.subheader("🔥 정규화된 클러스터별 평균 지표 (Z-score 기준)")

if not cluster_means.empty:
    scaler = StandardScaler()
    cluster_means_z = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        columns=cluster_means.columns,
        index=cluster_means.index
    )

    fig_hm_z, ax_hm_z = plt.subplots(figsize=(8, 4))
    sns.heatmap(cluster_means_z, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_hm_z)
    ax_hm_z.set_title("Normalized Heatmap by Cluster")
    plt.tight_layout()
    st.pyplot(fig_hm_z)
else:
    st.warning("선택된 클러스터 평균값이 없습니다.")

# ------------------------
# 8. Travel Safety 평가 (LLM 사용)
# ------------------------
st.header("✈️ 여행 안전성 평가 (Gemini AI 기반)")

user_country = st.text_input("여행하려는 국가명을 입력하세요:")

if user_country:
    country_data = df[df['Country'].str.lower() == user_country.lower()]
    
    if not country_data.empty:
        st.success(f"✅ {user_country}의 범죄 데이터를 찾았습니다.")
        st.dataframe(country_data)

        cluster_num = int(country_data.iloc[0]['Cluster'])
        indicators = {
            "살인": float(country_data['살인'].values[0]),
            "폭행": float(country_data['폭행'].values[0]),
            "아동납치": float(country_data['아동납치'].values[0]),
            "성폭력": float(country_data['성폭력'].values[0])
        }

        cluster_names = ["🟢 SafeZone", "🟠 CityRisk", "🔵 LethalZone", "🟣 NoGoZone"]
        cluster_explanations = {
            "🟢 SafeZone": "살인과 폭행 모두 낮은 안전한 지역입니다.",
            "🟠 CityRisk": "폭행률이 높으며, 도시 범죄 가능성이 있습니다.",
            "🔵 LethalZone": "살인률이 높은 위험 지역입니다.",
            "🟣 NoGoZone": "살인과 폭행 모두 매우 높은 고위험 지역입니다."
        }

        cluster_name = cluster_names[cluster_num]
        cluster_comment = cluster_explanations[cluster_name]

        prompt = f"""
여행자는 {user_country}로의 여행을 고려 중입니다.
이 나라는 클러스터 {cluster_num} - {cluster_name}에 속합니다.

범죄율 (인구 10만명당 기준):
- 살인: {indicators['Homicide']}
- 폭행: {indicators['Assault']}
- 아동납치: {indicators['Kidnapping']}
- 성폭력: {indicators['Sexual Violence']}

이 클러스터의 해석:
{cluster_comment}

여행 추천을 다음 중 하나로 제안해주세요:
→ "여행해도 안전합니다", "주의가 필요합니다", "여행을 권장하지 않습니다"
"""

        # ✅ Gemini 호출 함수
        def get_gemini_response(prompt: str) -> str:
            import google.generativeai as genai
            try:
                genai.configure(api_key="AIzaSyBSDINb61dWeRXN-6Ercawnl0ZBkUK7-j4")  # 🔑 여기에 본인의 Gemini API 키 입력
                model = genai.GenerativeModel("gemini-2.0-flash")  # ✅ 올바른 모델명
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"❗ Gemini 오류 발생: {e}"

        # AI 응답 출력
        st.markdown(f"### 🧠 클러스터 해석: {cluster_comment}")
        st.markdown("### 🤖 AI의 여행 안전성 평가 결과")
        result = get_gemini_response(prompt)
        st.write(result)

    else:
        st.warning("입력한 국가에 대한 데이터를 찾을 수 없습니다.")

# ------------------------
# 9. 국가별 지도 시각화
# ------------------------
st.subheader("🗺️ 국가별 클러스터 지도")
filtered_df['Cluster'] = filtered_df['Cluster'].astype(str)
color_map = {'0': '#1f77b4', '1': '#ff7f0e', '2': '#2ca02c', '3': '#d62728'}
fig_map = px.choropleth(
    filtered_df,
    locations='Country',
    locationmode='country names',
    color='Cluster',
    hover_name='Country',
    color_discrete_map=color_map
)
st.plotly_chart(fig_map, use_container_width=True)