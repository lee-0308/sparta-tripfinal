# app/pages/3_HotelSearch.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="호텔 요금 조회", page_icon="🏨", layout="centered")
st.title("🏨 호텔 요금 조회")

try:
    df_hotel = pd.read_csv("C:/sparta-tripfinal/Files/hotels_with_lalong.csv")
    st.success("호텔 데이터 로드 완료")
except FileNotFoundError:
    st.warning("호텔 데이터 파일이 없습니다.")
    df_hotel = pd.DataFrame()

selected_city = st.session_state.get("selected_city", "전체")

if selected_city == "전체":
    st.info("왼쪽에서 도시를 선택해 주세요.")
else:
    st.markdown(f"선택한 도시: **{selected_city}**")
    st.write("🏨 이 도시에 대한 호텔 정보가 아래 표시될 예정입니다.")
    st.dataframe(df_hotel[df_hotel["City"] == selected_city])
