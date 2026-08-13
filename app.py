# ============================================================
# STREAMLIT - PREDIKSI POPULARITAS APLIKASI GOOGLE PLAY STORE
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Prediksi Popularitas Aplikasi",
    page_icon="📱",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg,#ffffff 0%,#f7f8fc 100%); color:#17213b; }
.block-container { max-width:1500px; padding:2rem 2.5rem .5rem 2.5rem; }
[data-testid="stHeader"] { background:transparent; }
.main-title { text-align:center; color:#14203b; font-size:42px; font-weight:800; letter-spacing:-1.2px; margin:.1rem 0 .25rem; }
.subtitle { text-align:center; color:#697694; font-size:17px; margin-bottom:1.8rem; }
.metric-card { background:rgba(255,255,255,.97); border:1px solid #e8eaf1; border-radius:18px; min-height:105px; padding:20px 26px; box-shadow:0 5px 18px rgba(24,34,66,.07); display:flex; align-items:center; gap:18px; }
.metric-icon { width:58px; height:58px; border-radius:50%; background:linear-gradient(135deg,#f1e9ff,#f7f2ff); display:flex; align-items:center; justify-content:center; font-size:27px; flex-shrink:0; }
.metric-label { color:#56627d; font-size:14px; margin-bottom:3px; }
.metric-value { color:#5945d8; font-size:25px; font-weight:800; line-height:1.1; }
.input-card { background:rgba(255,255,255,.97); border:1px solid #e7e9f0; border-radius:20px; padding:25px; margin-top:22px; box-shadow:0 5px 18px rgba(24,34,66,.07); }

/* Satu card besar untuk seluruh area input */
.st-key-input_card {
    background:rgba(255,255,255,.98);
    border:1px solid #e1e5ee !important;
    border-radius:20px !important;
    padding:24px 24px 20px 24px !important;
    margin-top:22px;
    box-shadow:0 5px 18px rgba(24,34,66,.07);
}

/* Hilangkan jarak berlebihan di dalam card */
.st-key-input_card > div {
    gap:0.65rem;
}

/* Preview dan tombol tetap menjadi bagian dari card */
.st-key-input_card [data-testid="stExpander"] {
    margin-top:14px;
}

.st-key-input_card .stButton {
    margin-top:14px;
    margin-bottom:0;
}
.section-title { color:#17213b; font-size:22px; font-weight:800; margin-bottom:3px; }
.section-subtitle { color:#6d7891; font-size:14px; margin-bottom:18px; }
label,[data-testid="stWidgetLabel"] p { color:#1b2946 !important; font-weight:650 !important; }
div[data-baseweb="select"]>div { border:1px solid #dfe3ec !important; border-radius:11px !important; min-height:48px !important; background:#fff !important; }
div[data-baseweb="select"]>div:hover { border-color:#8d7be8 !important; }
input { border-radius:11px !important; }
.feature-box { background:linear-gradient(135deg,#eef7ff,#eaf3ff); border-radius:13px; border:1px solid #d9eaff; padding:17px 19px; margin-top:27px; color:#2167bd; }
.feature-title { font-weight:800; font-size:15px; margin-bottom:6px; }
.feature-text { font-size:14px; }
[data-testid="stExpander"] { border:1px solid #e0e4ec !important; border-radius:12px !important; background:#fff !important; margin-top:15px; }
.stButton { margin-top:16px; }
.stButton>button { width:100%; height:55px; border:none !important; border-radius:12px !important; background:linear-gradient(90deg,#7040d8 0%,#246fe8 100%) !important; color:#fff !important; font-size:16px !important; font-weight:800 !important; box-shadow:0 7px 18px rgba(75,76,210,.22); transition:.2s ease; }
.stButton>button:hover { transform:translateY(-1px); box-shadow:0 10px 22px rgba(75,76,210,.30); }
.result-popular,.result-not-popular { padding:24px; border-radius:16px; text-align:center; margin-top:15px; }
.result-popular { background:#ecfdf5; border:1px solid #b7efd5; }
.result-not-popular { background:#fff7ed; border:1px solid #fed7aa; }
.result-title { color:#66718b; font-size:14px; }
.result-value { font-size:30px; font-weight:800; margin-top:4px; }
.footer { text-align:center; color:#77819a; font-size:13px; padding:24px 0 8px; margin-top:20px; border-top:1px solid #e7e9ef; }
@media(max-width:800px){ .block-container{padding:1rem 1rem .5rem;} .main-title{font-size:30px;} .subtitle{font-size:14px;} .metric-card{margin-bottom:12px;} }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL DAN PREPROCESSING
# ============================================================

@st.cache_resource
def load_files():

    model = joblib.load("final_model.pkl")
    scaler = joblib.load("scaler.pkl")

    le_category = joblib.load("le_category.pkl")
    le_type = joblib.load("le_type.pkl")
    le_content = joblib.load("le_content_rating.pkl")
    le_popularity = joblib.load("le_popularity.pkl")

    return (
        model,
        scaler,
        le_category,
        le_type,
        le_content,
        le_popularity
    )


try:

    (
        model,
        scaler,
        le_category,
        le_type,
        le_content,
        le_popularity
    ) = load_files()

except Exception as e:

    st.error(
        "File model atau preprocessing tidak ditemukan. "
        "Pastikan seluruh file .pkl berada dalam folder yang sama dengan app.py."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📱 Prediksi Popularitas Aplikasi</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Prediksi popularitas aplikasi Google Play Store
        menggunakan model Random Forest
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INFORMASI MODEL
# ============================================================

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    <div class="metric-card"><div class="metric-icon">🧠</div><div>
    <div class="metric-label">Model</div><div class="metric-value">Random Forest</div>
    </div></div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown("""
    <div class="metric-card"><div class="metric-icon">🎯</div><div>
    <div class="metric-label">Accuracy</div><div class="metric-value">91.15%</div>
    </div></div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown("""
    <div class="metric-card"><div class="metric-icon">📊</div><div>
    <div class="metric-label">F1-Score Macro</div><div class="metric-value">0.7992</div>
    </div></div>
    """, unsafe_allow_html=True)

st.write("")


# ============================================================
# INPUT DATA - SATU CARD BESAR
# ============================================================

with st.container(border=True, key="input_card"):

    st.markdown(
        '<div class="section-title">📝 Masukkan Data Aplikasi</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-subtitle">Isi informasi aplikasi di bawah ini untuk melakukan prediksi.</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2, gap="large")

    # ========================================================
    # KOLOM KIRI
    # ========================================================

    with left:

        category = st.selectbox(
            "Category",
            options=list(le_category.classes_),
            help="Pilih kategori aplikasi."
        )

        rating = st.slider(
            "Rating",
            min_value=0.0,
            max_value=5.0,
            value=4.0,
            step=0.1,
            help="Masukkan rating aplikasi antara 0 sampai 5."
        )

        reviews = st.number_input(
            "Jumlah Reviews",
            min_value=0,
            value=1000,
            step=100,
            help="Masukkan jumlah ulasan aplikasi."
        )

    # ========================================================
    # KOLOM KANAN
    # ========================================================

    with right:

        app_type = st.selectbox(
            "Type",
            options=list(le_type.classes_),
            help="Pilih apakah aplikasi Free atau Paid."
        )

        content_rating = st.selectbox(
            "Content Rating",
            options=list(le_content.classes_),
            help="Pilih klasifikasi usia aplikasi."
        )

        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">ⓘ &nbsp; Fitur yang digunakan model</div>
            <div class="feature-text">Category, Rating, Reviews, Type, dan Content Rating.</div>
        </div>
        """, unsafe_allow_html=True)

    # ========================================================
    # PREVIEW INPUT
    # ========================================================

    st.write("")

    with st.expander("Lihat Data Input"):

        preview = pd.DataFrame({
            "Fitur": [
                "Category",
                "Rating",
                "Reviews",
                "Type",
                "Content Rating"
            ],
            "Nilai": [
                category,
                rating,
                reviews,
                app_type,
                content_rating
            ]
        })

        st.dataframe(
            preview,
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # TOMBOL PREDIKSI
    # ========================================================

    st.write("")

    predict_button = st.button(
        "🔍 Prediksi Popularitas",
        type="primary",
        use_container_width=True
    )


# ============================================================
# PROSES PREDIKSI
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # ENCODING
        # ----------------------------------------------------

        category_enc = le_category.transform(
            [category]
        )[0]

        type_enc = le_type.transform(
            [app_type]
        )[0]

        content_enc = le_content.transform(
            [content_rating]
        )[0]


        # ----------------------------------------------------
        # LOG TRANSFORMATION
        # ----------------------------------------------------

        reviews_log = np.log1p(reviews)


        # ----------------------------------------------------
        # URUTAN FITUR HARUS SAMA DENGAN TRAINING
        #
        # Category_enc
        # Rating
        # Reviews_log
        # Type_enc
        # ContentRat_enc
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [[
                category_enc,
                rating,
                reviews_log,
                type_enc,
                content_enc
            ]],
            columns=[
                "Category_enc",
                "Rating",
                "Reviews_log",
                "Type_enc",
                "ContentRat_enc"
            ]
        )


        # ----------------------------------------------------
        # SCALING
        # ----------------------------------------------------

        input_scaled = scaler.transform(
            input_data
        )


        # ----------------------------------------------------
        # PREDIKSI
        # ----------------------------------------------------

        prediction = model.predict(
            input_scaled
        )[0]


        # Mengubah label numerik menjadi teks
        prediction_label = le_popularity.inverse_transform(
            [prediction]
        )[0]


        # ----------------------------------------------------
        # PROBABILITAS
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            input_scaled
        )[0]

        class_labels = le_popularity.inverse_transform(
            model.classes_.astype(int)
        )

        probability_dict = dict(
            zip(
                class_labels,
                probabilities
            )
        )


        # ====================================================
        # HASIL PREDIKSI
        # ====================================================

        st.write("")
        st.divider()

        st.subheader("Hasil Prediksi")


        if prediction_label == "Populer":
            st.markdown(
            '<div class="result-popular">'
            '<div class="result-title">Hasil Prediksi</div>'
            '<div class="result-value">POPULER</div>'
            '</div>',
        unsafe_allow_html=True
        )

        else:
            st.markdown(
            '<div class="result-not-popular">'
            '<div class="result-title">Hasil Prediksi</div>'
            '<div class="result-value">TIDAK POPULER</div>'
            '</div>',
            unsafe_allow_html=True
        )

        # ====================================================
        # PROBABILITAS
        # ====================================================

        st.write("")

        p1, p2 = st.columns(2)

        with p1:

            popular_probability = (
                probability_dict.get("Populer", 0) * 100
            )

            st.metric(
                "Probabilitas Populer",
                f"{popular_probability:.2f}%"
            )


        with p2:

            not_popular_probability = (
                probability_dict.get(
                    "Tidak Populer",
                    0
                ) * 100
            )

            st.metric(
                "Probabilitas Tidak Populer",
                f"{not_popular_probability:.2f}%"
            )


        # ====================================================
        # GRAFIK PROBABILITAS PREDIKSI
        # ====================================================

        st.write("")
        st.subheader("Probabilitas Prediksi")

        prob_df = pd.DataFrame({
            "Kelas": ["Populer", "Tidak Populer"],
            "Probabilitas": [
                popular_probability,
                not_popular_probability
            ]
        })

        fig = px.bar(
            prob_df,
            x="Probabilitas",
            y="Kelas",
            orientation="h",
            text="Probabilitas"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="inside"
        )

        fig.update_layout(
            xaxis_title="Probabilitas (%)",
            yaxis_title="",
            xaxis=dict(range=[0, 100]),
            showlegend=False,
            height=300,
            margin=dict(l=10, r=10, t=20, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
        
    except Exception as e:
    
        st.error(
            f"Terjadi kesalahan saat melakukan prediksi: {e}"
        )
# ============================================================
# FOOTER
# ============================================================

st.write("")
st.divider()

st.caption(
    "Sistem Prediksi Popularitas Aplikasi Google Play Store "
    "• Bintang Ragielfah Herdiman Syahbilillah"
)