# app/main.py
import streamlit as st

st.set_page_config(
    page_title="Trip Planner",
    page_icon="🧳",
    layout="centered"
)

st.title("✈️ 여행 계획 프로토타입")
st.markdown("""
### 즐거운 해외여행, 열심히 노력한 당신은 이제 떠날 시간입니다.
            
### 어디로 갈지 정하셨나요?

왼쪽 사이드바에서 아래 항목 중 하나를 선택해 주세요:

- 🌏 여행지 찾기
- 🛫 항공권 검색
- 🏨 호텔 검색
- 💬 리뷰 기반 도시 탐색  
""")
