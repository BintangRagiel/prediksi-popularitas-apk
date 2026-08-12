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

/* Background */
.stApp {
    background-color: #EDEFF3;
}

/* Judul utama */
.main-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #6b7280;
    margin-bottom: 30px;
}

/* Card */
.card {
    background: white;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}

/* Result */
.result-popular {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
}

.result-not-popular {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
}

.result-title {
    font-size: 16px;
    color: #6b7280;
}

.result-value {
    font-size: 32px;
    font-weight: 700;
    margin-top: 5px;
}

/* Tombol */
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 16px;
    background-color: #4F46E5;
    color: white;
    border: none;
}

.stButton > button:hover {
    background-color: #4338CA;
    color: white;
}

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

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Model",
        value="Random Forest"
    )

with col2:
    st.metric(
        label="Accuracy",
        value="91.06%"
    )

with col3:
    st.metric(
        label="F1-Score Macro",
        value="0.7991"
    )


st.write("")


# ============================================================
# INPUT DATA
# ============================================================

st.subheader("Masukkan Data Aplikasi")

st.caption(
    "Isi informasi aplikasi di bawah ini untuk melakukan prediksi."
)

left, right = st.columns(2)


# ============================================================
# KOLOM KIRI
# ============================================================

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


# ============================================================
# KOLOM KANAN
# ============================================================

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

    st.info(
        """
        **Fitur yang digunakan model**

        Category, Rating, Reviews, Type, dan Content Rating.
        """
    )


# ============================================================
# PREVIEW INPUT
# ============================================================

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


# ============================================================
# TOMBOL PREDIKSI
# ============================================================

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