import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from PIL import Image
from pathlib import Path

# 페이지 설정
st.set_page_config(layout="wide", page_title="여행지 검색", page_icon="🌏")

# 세션 상태 초기화
if "selected_city" not in st.session_state:
    st.session_state.selected_city = "전체"
if "map_clicked" not in st.session_state:
    st.session_state.map_clicked = False
if "show_spots" not in st.session_state:
    st.session_state.show_spots = True
if "show_hotels" not in st.session_state:
    st.session_state.show_hotels = True
if "show_airports" not in st.session_state:
    st.session_state.show_airports = True
if "selected_place_info" not in st.session_state:
    st.session_state.selected_place_info = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Files"
IMG_DIR = BASE_DIR / "Images"

# 이미지 경로 함수
def get_flag_path(city, country):
    exceptions = ["Hong Kong", "Macau"]
    key = city if city in exceptions else country
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".avif"]:
        path = IMG_DIR / "Flags" / f"{key}{ext}"
        if path.exists():
            return path
    return None

def find_image(city_name, place_name):
    if not isinstance(place_name, str):
        place_name = str(place_name) if pd.notna(place_name) else ""
    safe_city = city_name.replace("/", "_").replace("\\", "_").strip()
    safe_place = place_name.replace("/", "_").replace("\\", "_").strip()
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".avif"]:
        path = IMG_DIR / "Spots" / f"{safe_city}_{safe_place}{ext}"
        if path.exists():
            return path
    return None

# 데이터 로드
df_city = pd.read_csv(DATA_DIR /"cities_bbox.csv")
df_review = pd.read_csv(DATA_DIR /"review_and_fee.csv")
df_info = pd.read_csv(DATA_DIR /"worldcities_cost_fin.csv")
df_spot = pd.read_csv(DATA_DIR /"revie_spots_with_coords.csv")
df_airport = pd.read_csv(DATA_DIR /"airports_full_done.csv")
df_hotel = pd.read_csv(DATA_DIR /"hotels_with_lalong.csv")
df_flight = pd.read_csv(DATA_DIR /"LCC_FSC_Hour_Price_USD.csv")

avg_rating = df_review.groupby("SpotName")["ReviewRating"].mean().reset_index()
avg_rating.columns = ["city", "avg_rating"]
df_flight["Avg_Flight_Hour"] = df_flight[["In_Hour", "Out_Hour"]].mean(axis=1)

merged_info = pd.merge(df_info, df_city, left_on='City_name', right_on='city', how='left')

# --- 사이드바 ---
with st.sidebar:
    city_options = ["전체"] + sorted(df_city["city"].unique())

    selected_city = st.selectbox(
        "도시 선택",
        city_options,
        index=city_options.index(st.session_state.selected_city)
        if st.session_state.selected_city in city_options else 0
    )

    # 선택 변경 감지 → 상태 업데이트
    if selected_city != st.session_state.selected_city:
        st.session_state.selected_city = selected_city
        st.session_state.map_clicked = False
        st.session_state.selected_place_info = None
        # 체크박스 값 초기화
        st.session_state.show_spots = True
        st.session_state.show_hotels = True
        st.session_state.show_airports = True
        st.rerun()

    # 체크박스 UI
    if selected_city != "전체":
        st.checkbox("관광지", key="show_spots")
        st.checkbox("호텔", key="show_hotels")
        st.checkbox("공항", key="show_airports")
    else:
        st.session_state.show_spots = False
        st.session_state.show_hotels = False
        st.session_state.show_airports = False

# --- 전체 도시 지도 ---
if selected_city == "전체":
    st.markdown("### 여행하고 싶은 곳을 선택하세요.")
    st.markdown("### 지도를 클릭하거나 필터에서 선택하세요.")
    m = folium.Map(location=[df_city["latitude"].mean(), df_city["longitude"].mean()], zoom_start=2)

    for _, row in df_city.iterrows():
        folium.Marker([row["latitude"], row["longitude"]],
                      popup=row["city"], tooltip=row["city"],
                      icon=folium.Icon(color="blue")).add_to(m)

    map_data = st_folium(m, width=1200, height=700)

    if map_data and map_data.get("last_clicked") and not st.session_state.map_clicked:
        click_lat = map_data["last_clicked"]["lat"]
        click_lon = map_data["last_clicked"]["lng"]
        df_city["distance"] = np.sqrt((df_city["latitude"] - click_lat) ** 2 + (df_city["longitude"] - click_lon) ** 2)
        nearest_city = df_city.loc[df_city["distance"].idxmin()]
        if nearest_city["distance"] < 2:
            st.session_state.selected_city = nearest_city["city"]
            st.session_state.map_clicked = True
            st.session_state.selected_place_info = None
            st.rerun()

# --- 도시 상세 보기 ---
else:
    st.session_state.map_clicked = False
    col1, col2, col3 = st.columns([1.2, 2.5, 1.3])
    city_df = df_city[df_city["city"] == selected_city]
    lat, lon = city_df["latitude"].values[0], city_df["longitude"].values[0]

    with col1:
        st.subheader("상세 정보")
        place = st.session_state.selected_place_info
        if place:
            name = next((place.get(k) for k in ["SpotName", "Hotel Name", "Airport_Name"] if pd.notna(place.get(k))), "이름 정보 없음")
            icon = "📍" if "SpotName" in place else ("🏨" if "Hotel Name" in place else "🛫")
            st.markdown(f"{icon} **{name}**")

            img_path = find_image(selected_city, name)
            if img_path:
                st.image(Image.open(img_path), use_container_width=True)
            else:
                st.info("사진이 없습니다.")

            if "SpotName" in place:
                reviews = df_review[df_review["SpotName"] == name]["ReviewText"].dropna().tolist()
            elif "Hotel Name" in place:
                reviews = df_review[df_review["HotelName"] == name]["ReviewText"].dropna().tolist() if "HotelName" in df_review.columns else []
            else:
                reviews = []

            if reviews:
                st.markdown("**리뷰 예시:**")
                for rev in reviews[:3]:
                    st.write(f"- {rev}")
            else:
                st.info("리뷰가 없습니다.")
        else:
            st.info("지도 마커를 클릭해보세요!")

    with col2:
        m = folium.Map(location=[lat, lon], zoom_start=10)
        folium.Marker([lat, lon], tooltip=selected_city, icon=folium.Icon(color="blue")).add_to(m)

        def add_markers(df, lat_col, lon_col, name_col, color, icon, city_field="City"):
            for _, row in df[df[city_field] == selected_city].iterrows():
                folium.Marker([row[lat_col], row[lon_col]],
                              popup=row[name_col],
                              icon=folium.Icon(color=color, icon=icon)).add_to(m)

        if st.session_state.show_spots:
            add_markers(df_spot, "Latitude", "Longitude", "SpotName", "green", "flag")
        if st.session_state.show_hotels:
            add_markers(df_hotel, "Latitude", "Longitude", "Hotel Name", "red", "home")
        if st.session_state.show_airports:
            add_markers(df_airport, "Latitude", "Longitude", "Airport_Name", "black", "plane", "City_name")

        map_data = st_folium(m, width=900, height=700)

        if map_data and map_data.get("last_clicked"):
            click_lat, click_lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]

            def compute_distance(df, lat_col, lon_col):
                return np.sqrt((df[lat_col] - click_lat) ** 2 + (df[lon_col] - click_lon) ** 2)

            candidates = []
            if st.session_state.show_spots:
                df_tmp = df_spot[df_spot["City"] == selected_city].copy()
                df_tmp["distance"] = compute_distance(df_tmp, "Latitude", "Longitude")
                df_tmp["type"] = "spot"
                candidates.append(df_tmp)
            if st.session_state.show_hotels:
                df_tmp = df_hotel[df_hotel["City"] == selected_city].copy()
                df_tmp["distance"] = compute_distance(df_tmp, "Latitude", "Longitude")
                df_tmp["type"] = "hotel"
                candidates.append(df_tmp)
            if st.session_state.show_airports:
                df_tmp = df_airport[df_airport["City_name"] == selected_city].copy()
                df_tmp["distance"] = compute_distance(df_tmp, "Latitude", "Longitude")
                df_tmp["type"] = "airport"
                candidates.append(df_tmp)

            if candidates:
                all_places = pd.concat(candidates, ignore_index=True)
                nearest = all_places.loc[all_places["distance"].idxmin()]
                if nearest["distance"] < 0.5:
                    st.session_state.selected_place_info = nearest.to_dict()
                    st.session_state.selected_place = nearest.get("SpotName") or nearest.get("Hotel Name") or nearest.get("Airport_Name")
                    st.session_state.selected_place_type = nearest["type"]
                else:
                    st.session_state.selected_place_info = None

    with col3:
        country = merged_info.loc[merged_info["City_name"] == selected_city, "Country"].values
        country = country[0] if len(country) > 0 else None
        flag_path = get_flag_path(selected_city, country) if country else None
        if flag_path:
            st.image(Image.open(flag_path), width=100)
        else:
            st.info("국기 이미지가 없습니다.")

        st.subheader(f"{selected_city} 정보")
        row = df_info[df_info["City_name"] == selected_city]
        if not row.empty:
            r = row.iloc[0]
            st.markdown(f"**면적:** {r['Area(square_km)']} km²")
            st.markdown(f"**인구:** {r['Population(thousands)']}천 명")
            st.markdown(f"**점심 (USD):** ${float(r['avg_lunch_USD']):.2f}" if pd.notna(r['avg_lunch_USD']) else f"**점심 (USD):** {r['avg_lunch_USD']}")
            st.markdown(f"**저녁 (USD):** ${float(r['avg_supper_USD']):.2f}" if pd.notna(r['avg_supper_USD']) else f"**저녁 (USD):** {r['avg_supper_USD']}")
            st.markdown("**공항 → 도심 이동**")
            st.markdown(f"- 소요 시간: {r['time_from_Airport_to_Downtown']}")
            st.markdown(f"- 요금: ${float(r['A_to_D_USD']):.2f}" if pd.notna(r['A_to_D_USD']) else f"- 요금: {r['A_to_D_USD']}")
        else:
            st.warning("도시 정보가 없습니다.")
