import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    layout="centered",  
    page_title="항공권 검색 및 도착 공항 지도",
    page_icon="✈️"
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "Files"
IMG_DIR = BASE_DIR / "Images"

# 1. 데이터 불러오기
df = pd.read_csv(DATA_DIR / "final_flights.csv")

# 2. 날짜 형식 변환
df['Departure_Date'] = pd.to_datetime(df['Departure_Date'], errors='coerce')

df = df[df['Departure_Date'].notnull()]


# 3. Streamlit 타이틀
st.title("✈️ 항공권 검색 및 도착 공항 지도")

# 4. 출발 공항 선택
departure_airports = df['Dep_Airport_KOR'].dropna().unique().tolist()
selected_airport = st.session_state.get('selected_airport', departure_airports[0])
departure_airports = sorted(departure_airports)
index = departure_airports.index(selected_airport) if selected_airport in departure_airports else 0
print(index)
selected_departure = st.selectbox("출발 공항을 선택하세요", departure_airports, index=index)


# 5. 도착 공항 필터 (출발지 기준)
filtered_df1 = df[df['Dep_Airport_KOR'] == selected_departure]
arrival_airports = filtered_df1['Arriv_Airport_KOR'].dropna().unique()
selected_arrival = st.selectbox("도착 공항을 선택하세요", sorted(arrival_airports))

# 6. 항공사 선택
filtered_df2 = filtered_df1[filtered_df1['Arriv_Airport_KOR'] == selected_arrival]
airlines = filtered_df2['Airline'].dropna().unique()
selected_airline = st.selectbox("항공사를 선택하세요", ["전체 보기"] + sorted(airlines.tolist()))

# 슬라이더 범위 설정용 최소/최대값
min_date = df['Departure_Date'].min().to_pydatetime()
max_date = df['Departure_Date'].max().to_pydatetime()

# Streamlit 슬라이더
selected_date_range = st.slider(
    "출발일 범위 선택",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

# 8. 필터링된 데이터
filtered = df[
    (df['Dep_Airport_KOR'] == selected_departure) &
    (df['Arriv_Airport_KOR'] == selected_arrival) &
    (df['Departure_Date'] >= selected_date_range[0]) &
    (df['Departure_Date'] <= selected_date_range[1])
]

if selected_airline != "전체 보기":
    filtered = filtered[filtered['Airline'] == selected_airline]

# 9. 결과 테이블 표시
st.subheader("📝 조건에 맞는 항공편 리스트")
st.dataframe(filtered, use_container_width=True)

# 10. 지도에 도착 공항 표시
st.subheader("🗺️ 도착 공항 지도 표시")

# 도착 공항 기준으로 unique 값 추출
map_df = filtered.dropna(subset=['Arriv_Airport_Lat', 'Arriv_Airport_Long']).copy()

if map_df.empty:
    st.warning("선택한 조건에 해당하는 항공편이 없습니다.")
else:
    fig = px.scatter_geo(
        map_df,
        lat='Arriv_Airport_Lat',
        lon='Arriv_Airport_Long',
        text='Arriv_Airport_KOR',
        hover_name='Arriv_Airport_KOR',
        color='Airline',
        projection='natural earth',
        title="도착 공항 위치 (선택한 조건 기준)"
    )
    fig.update_layout(geo=dict(showland=True))
    st.plotly_chart(fig, use_container_width=True)

