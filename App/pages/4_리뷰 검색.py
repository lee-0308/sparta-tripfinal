# app/pages/3_ReviewSearch.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="리뷰 기반 탐색", page_icon="💬", layout="centered")
st.title("💬 리뷰 기반 도시 탐색")

try:
    df_review = pd.read_csv("C:/sparta-tripfinal/Files/review_and_fee.csv")
except:
    st.warning("리뷰 데이터가 존재하지 않습니다.")

st.info("이 페이지에서는 사용자 리뷰를 바탕으로 도시의 매력을 탐색할 수 있습니다.")
st.markdown("- 클러스터링, 감정 분석, 키워드 기반 필터 등이 여기에 구현될 예정입니다.")
