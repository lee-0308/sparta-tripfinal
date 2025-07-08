import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pydeck as pdk
from collections import Counter

st.set_page_config(
    layout="wide",  # 기본 'centered' → 'wide'로 변경
    page_title="Review Dashboard",
    page_icon="🧳"
)

# 데이터 로드
df_city = pd.read_csv("C:/sparta-tripfinal/Files/cities_bbox.csv")
df_review = pd.read_csv("C:/sparta-tripfinal/Files/review_and_fee.csv")
df_c1 = pd.read_csv("C:/sparta-tripfinal/Files/city_type_1.csv")
df_c2 = pd.read_csv("C:/sparta-tripfinal/Files/city_type_2.csv")
df_c3 = pd.read_csv("C:/sparta-tripfinal/Files/city_type_3.csv")
df_c4 = pd.read_csv("C:/sparta-tripfinal/Files/city_type_4.csv")
df_summary = pd.read_csv("C:/sparta-tripfinal/Files/cluster_sentiment_summary.csv")
df_count = pd.read_csv("C:/sparta-tripfinal/Files/cluster_spot_counts.csv")
df_pca = pd.read_csv("C:/sparta-tripfinal/Files/X_pca.csv")
df_tsne = pd.read_csv("C:/sparta-tripfinal/Files/X_tsne.csv")
df_umap = pd.read_csv("C:/sparta-tripfinal/Files/X_umap.csv")
df_spot = pd.read_csv("C:/sparta-tripfinal/Files/revie_spots_with_coords.csv")

cluster_labels = {
    "도심 랜드마크형": 1,
    "자연·감성 혼합형": 2,
    "전통유산 탐방형": 3,
    "관광 엔터테인먼트형": 4
}

df_c1['cluster'] = 0
df_c2['cluster'] = 1
df_c3['cluster'] = 2
df_c4['cluster'] = 3

df_city_cluster = pd.concat([
    df_c1[['City_name', 'cluster']],
    df_c2[['City_name', 'cluster']],
    df_c3[['City_name', 'cluster']],
    df_c4[['City_name', 'cluster']]
], ignore_index=True)

# 리뷰 데이터에 cluster 컬럼 병합
df_review = df_review.merge(df_city_cluster, on='City_name', how='left')
df_review['cluster'] = df_review['cluster'].fillna(-1).astype(int)

# 클러스터별 도시 조합
type_map = {1: df_c1, 2: df_c2, 3: df_c3, 4: df_c4}

# 필터 라벨
type_labels = {"전체": 0, "Cluster 0": 1, "Cluster 1": 2, "Cluster 2": 3, "Cluster 3": 4}
sentiment_options = ["전체"] + sorted(df_review["sentiment"].unique())

def autopct_format(pct):
    return f'{pct:.1f}%' if pct > 3 else ''

def draw_donut_chart(sizes, labels, title = "Sentiment Distribution"):
    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        autopct=autopct_format,
        startangle=140,
        colors=['#d62728', '#ff7f0e', '#2ca02c'],
        wedgeprops={'width': 0.4},
        pctdistance=0.75
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
    ax.axis('equal')
    ax.set_title(title, fontsize=14)
    ax.legend(wedges, labels, title="Sentiment", loc="center left", bbox_to_anchor=(1, 0.5))
    st.pyplot(fig)

# 불용어 목록
stop_words = {"the", "a", "an", "I", "my", "me", "mine", "you", "your", "yours",
              "he", "his", "him", "she", "her", "hers", "they", "them", "in",
              "of", "can", "for", "and", "it", "was", "to", "this", "as", "some",
              "through", "with", "that", "from", "is", "its", "there", "on", "get", 
              "at", "There", "into", "by", "are", "were", "s", "but", "visit", 
              "one", "so", "if", "or", "here", "not", "just", "we", "be", "have", "i",
              "most", "even", "had", "where", "has", "much", "both", "while", "our",
              "over", "do", "must", "than", "spend", "go", "bit", "how", "place", "see",
              "take", "no", "place", "see", "area", "too", "u", "which", "could", "very",
              "more", "any", "also", "been", "because", "sure", "when", "other", "went", 
              "before", "those", "will", "all", "would", "what", "better", "like", "park", 
              "felt", "feel", "such", "about", "only", "make", "around", "way", "quite",
              "almost", "us", "itself", "staff", "still", "did", "maybe", "after"
              }

positive_words = {"beautiful", "amazing", "worth", "love", "breathtaking", "great", "enjoy", "fun", 
                  "best", "nice", "recommend", "good", "impressive", "perfect", "cool"}
negative_words = {"boring", "terrible", "bad", "waste", "crowded", "dirty", "disappointing", "worst", "expensive",
                  "hate", "slow", "worst", "nothing", "sad", "wait", "far"}

def generate_wordcloud(text, title=None):
    if not text or len(text.strip()) == 0:
        st.write("워드 클라우드를 생성할 단어가 없습니다.")
        return
    try:
        if selected_sentiment != "전체":
            # 감정이 선택된 경우: 가중치 적용
            words = [w.lower() for w in text.split() if w.isalpha() and w.lower() not in stop_words]
            word_freq = Counter(words)

            if selected_sentiment.lower() == "positive":
                for word in word_freq:
                    if word in positive_words:
                        word_freq[word] *= 3
            elif selected_sentiment.lower() == "negative":
                for word in word_freq:
                    if word in negative_words:
                        word_freq[word] *= 5
                    elif word in positive_words:
                        word_freq[word] = int(word_freq[word]* 0.5)

            wordcloud = WordCloud(
                background_color='lightgray',
                max_words=100,
                stopwords=stop_words,
                width=800,
                height=400,
                colormap='viridis'
            ).generate_from_frequencies(word_freq)

        else:
            # 전체일 경우: 텍스트 그대로
            wordcloud = WordCloud(
                background_color='lightgray',
                max_words=100,
                stopwords=stop_words,
                width=800,
                height=400,
                colormap='viridis'
            ).generate(text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    except ValueError:
        st.write("워드클라우드를 생성할 충분한 단어가 없습니다.")

if "selected_cluster" not in st.session_state:
    st.session_state.selected_cluster = "전체"

if "selected_city" not in st.session_state:
    st.session_state.selected_city = "전체"

if "selected_sentiment" not in st.session_state:
    st.session_state.selected_sentiment = "전체"

# 사이드바
with st.sidebar:
    st.title("리뷰 대시보드")

    # 군집 필터
    selected_cluster = st.selectbox("**클러스터 선택**", type_labels)
    type_num = type_labels[selected_cluster]
    
    # 도시 필터
    # 클러스터에 따라 도시 필터링
    if type_num == 0:
        city_options = ["전체"]
    else:
        city_options = ["전체"] + sorted(type_map[type_num]["City_name"].dropna().unique())
    # 도시 선택 박스
    selected_city = st.selectbox("**도시 선택**", city_options,)
    # 감정 필터
    selected_sentiment = st.selectbox("**감정 선택**", sentiment_options )

# 필터 적용
df_filtered = df_review.copy()
if selected_city != "전체":
    df_filtered = df_filtered[df_filtered["City_name"] == selected_city]
if selected_sentiment != "전체":
    df_filtered = df_filtered[df_filtered["sentiment"] == selected_sentiment]
if type_num != 0:
    df_filtered = df_filtered[df_filtered["cluster"] == (type_num - 1)]

with st.container():
    st.markdown("### 클러스터 설명")

    data = pd.DataFrame({
    "클러스터": [
        "Cluster 0",
        "Cluster 1",
        "Cluster 2",
        "Cluster 3"
    ],
    "주요 특징 키워드": [
        "Museum, Palace, Cultural Sites",
        "Park, Photo Spot, Nature + Emotion",
        "Heritage, Zoo, Historic Street",
        "Theme Park, Tower, Urban Attractions"
    ],
    "대표 관광지 예시": [
        "Emirates Heritage Village",
        "Ueno Park, Gyeongbokgung Plaza",
        "Kyoto Old Town",
        "Tokyo Tower, Universal Studios"
    ],
    "관광지 유형": [
        "도심 랜드마크형",
        "자연·감성 혼합형",
        "전통유산 탐방형",
        "관광 엔터테인먼트형"
    ]
    })

    # 전체 선택 시 전체 표 출력, 특정 클러스터 선택 시 해당 행만 출력
    if selected_cluster == "전체":
        display_data = data
    else:
        display_data = data[data["클러스터"] == selected_cluster]

    # 스타일링
    css = """
    <style>
    .styled-table-wrapper {
        width: 100%;
        overflow-x: auto;
    }
    table.styled-table {
        border-collapse: collapse;
        font-size: 13px;
        font-family: 'Arial', sans-serif;
        width: 100%;
        border: 1px solid #ddd;
        box-shadow: 0 0 5px rgba(0,0,0,0.1);
    }
    table.styled-table thead tr {
        background-color: #cccccc;
        color: #000000;
        text-align: center;
    }
    table.styled-table th,
    table.styled-table td {
        padding: 6px 8px;
        text-align: center;
    }
    table.styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    /* 선택한 유형 행에 빨간색 테두리 */
    .highlight-row {
        border: 2px solid red !important;
    }
    </style>
    """

    # HTML 테이블 생성
    html_table = "<div class='styled-table-wrapper'><table class='styled-table'>"
    html_table += "<thead><tr>"
    for col in data.columns:
        html_table += f"<th>{col}</th>"
    html_table += "</tr></thead><tbody>"

    for _, row in data.iterrows():
        # 선택한 유형이면 highlight-row 클래스 추가
        highlight_class = "highlight-row" if row["클러스터"] == selected_cluster else ""
        html_table += f"<tr class='{highlight_class}'>"
        for item in row:
            html_table += f"<td>{item}</td>"
        html_table += "</tr>"

    html_table += "</tbody></table></div>"

    st.markdown(css + html_table, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.container():
    # 클러스터별 관광지 분포
    st.markdown("### 클러스터별 관광지 분포")

    label_lookup = {v-1: k for k, v in cluster_labels.items() if v != 0}

    # 클러스터 라벨 영어 매핑
    cluster_label_en = {
        0: "Overall",
        1: "Urban Landmark",
        2: "Nature & Emotion",
        3: "Heritage Tour",
        4: "Entertainment"
    }
    # 클러스터 번호를 이름으로 변환
    df_count_named = df_count.copy()
    df_count_named["cluster"] = df_count_named["cluster"].map(lambda x: cluster_label_en.get(x+1, "Unknown"))

    figsize = (16, 3)  # 가로 넓게, 세로는 줄이기
    annot_kws = {"fontsize":11, "fontweight":"bold"}  # 숫자 크기 및 굵기
    xtick_rotation = 35
    xtick_fontsize = 11

    if type_num == 0:
        # 전체 분포용 데이터 생성
        data_for_heatmap = df_count.drop(columns=["cluster"]).sum().to_frame().T
        data_for_heatmap.index = [""]

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            data_for_heatmap,
            annot=True,
            fmt="d",
            cmap="YlOrBr",
            linewidths=0.4,
            ax=ax,
            annot_kws=annot_kws,
        )
        ax.set_title("Overall", fontsize=15, pad=12)
        ax.set_xlabel("", fontsize=12, labelpad=10)
        ax.set_ylabel("")
        ax.tick_params(axis='x', rotation=xtick_rotation, labelsize=xtick_fontsize)
        st.pyplot(fig)

    else:
        selected_cluster_en = cluster_label_en.get(type_num, "Unknown")
        data_for_heatmap = df_count_named[df_count_named["cluster"] == selected_cluster_en]
        data_for_heatmap = data_for_heatmap.set_index("cluster")
        data_for_heatmap.index = [""]

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            data_for_heatmap,
            annot=True,
            fmt="d",
            cmap="YlOrBr",
            linewidths=0.4,
            ax=ax,
            annot_kws=annot_kws,
        )
        ax.set_title(f"Tourist Spot Distribution - {selected_cluster_en}", fontsize=15, pad=12)
        ax.set_xlabel("", fontsize=12, labelpad=10)
        ax.set_ylabel("")
        ax.tick_params(axis='x', rotation=xtick_rotation, labelsize=xtick_fontsize)
        st.pyplot(fig)

st.markdown("<br>", unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns([1,2])
    with col1:
        # 감정분포 도넛차트
        st.markdown("### 감정 분포 차트")
        # 감정 차트용 데이터: 감정 필터 제외
        df_donut = df_review.copy()
        if selected_city != "전체":
            df_donut = df_donut[df_donut["City_name"] == selected_city]
        if type_num != 0:
            df_donut = df_donut[df_donut["cluster"] == (type_num - 1)]

        sentiment_counts = df_donut["sentiment"].value_counts()
        labels = ['Negative', 'Neutral', 'Positive']
        sizes = [sentiment_counts.get(label.lower(), 0) for label in labels]

        if sum(sizes) > 0:
            title = f"Sentiment Distribution - {selected_city}" if selected_city != "전체" else "Sentiment Distribution"
            draw_donut_chart(sizes, labels, title)
        else:
            st.write("해당 조건에 감정 데이터가 없습니다.")

    with col2:
        st.markdown("### 워드 클라우드")

        text_data = " ".join(df_filtered["ReviewText"].dropna())
        words = [w.lower() for w in text_data.split() if w.isalpha() and w.lower() not in stop_words]
        cleaned_text = " ".join(words)
        title = selected_city
        if selected_sentiment != "전체":
            title += f" / {selected_sentiment}"
        if selected_cluster != "전체":
            title += f" / {selected_cluster}"
        generate_wordcloud(text_data, title)

st.markdown("<br>", unsafe_allow_html=True)

with st.container():
    st.markdown("### 지도")

    cluster_colors = {
        0: [0, 0, 0],
        1: [255, 99, 132],
        2: [54, 162, 235],
        3: [255, 206, 86],
        4: [75, 192, 192],
    }

    if selected_city != "전체":
        city_info = df_city[df_city["city"] == selected_city]
        if not city_info.empty:
            lat = city_info.iloc[0]["latitude"]
            lon = city_info.iloc[0]["longitude"]
            city_cluster = 0
            for cid, cdf in type_map.items():
                if selected_city in cdf["City_name"].values:
                    city_cluster = cid
                    break
            df_map = pd.DataFrame([{"city": selected_city, "lat": lat, "lon": lon, "cluster": city_cluster}])
        else:
            df_map = pd.DataFrame(columns=["city", "lat", "lon", "cluster"])
    else:
        if type_num == 0:
            df_map = df_city.rename(columns={"latitude": "lat", "longitude": "lon"}).copy()
            df_map["cluster"] = 0
        else:
            cluster_df = type_map[type_num]
            df_map = cluster_df.merge(df_city, left_on="City_name", right_on="city", how="left")
            df_map = df_map.rename(columns={"latitude": "lat", "longitude": "lon"})
            df_map["cluster"] = type_num

    df_map["color"] = df_map["cluster"].apply(lambda x: cluster_colors.get(x, [100, 100, 100]))
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position='[lon, lat]',
        get_radius=30000,
        get_fill_color='[color[0], color[1], color[2]]',
        pickable=True,
        radius_min_pixels=5,
        radius_max_pixels=10,
    )
    mid_lat = df_map["lat"].mean() if not df_map.empty else 30
    mid_lon = df_map["lon"].mean() if not df_map.empty else 0
    view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=2.5, pitch=0, bearing=0)
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="light"))
