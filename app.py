import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import gdown
import requests
import json
import base64
from io import BytesIO
import re

# =========================
# 🔧 KONFIGURASI UI & GAYA
# =========================
APP_TITLE = "Deteksi Penyakit Daun Jagung"
APP_ICON = "🌽"
BACKGROUND_COLOR_A = "#89A827"
BACKGROUND_COLOR_B = "#DFD730"
BACKGROUND_COLOR_C = "#FFEE48"
BACKGROUND_COLOR_D = "#FDFDB6"
BACKGROUND_COLOR_E = "#F8F9FA"
TITLE_COLOR = "#89A827"
TEXT_COLOR = "#262730"

def apply_custom_styles():
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(to bottom, 
            {BACKGROUND_COLOR_A} 0%,
            {BACKGROUND_COLOR_B} 5%,
            {BACKGROUND_COLOR_C} 8%,
            {BACKGROUND_COLOR_D} 15%,
            {BACKGROUND_COLOR_E} 30%,
            {BACKGROUND_COLOR_E} 100%
        );
        min-height: 100vh;
        display: flex;
        flex-direction: column;
    }}

    .main {{
        flex-grow: 1;
    }}

    footer {{
        margin-top: auto;
        padding: 16px;
        text-align: center;
        font-size: 1em;
        color: {TEXT_COLOR};
    }}
    /* Gaya untuk mengatur lebar maksimum gambar yang diunggah */
    .stApp img {{
        max-width: 100%; /* Pastikan gambar tidak melebihi lebar kolomnya */
        height: auto; /* Biarkan tinggi menyesuaikan agar aspek rasio terjaga */
        object-fit: contain; /* Pastikan seluruh gambar terlihat tanpa terpotong */
    }}
    /* Jika Anda ingin spesifik hanya gambar di kolom prediksi */
    .image-display-box img {{
        width: 100%; 
        max-height: 450px; 
        object-fit: contain;
    }}
    /* Menyesuaikan lebar header agar konsisten (opsional, jika dirasa perlu) */
    .stApp .css-1lcbmhc {{ /* Ini class untuk main content container */
        max-width: 1200px; /* Contoh: batasi lebar total aplikasi */
        margin: auto; /* Pusatkan */
    }}
    .stApp .reportview-container .main .block-container {{
        padding-top: 1rem; 
        padding-bottom: 1rem; 
    }}
    </style>
    """, unsafe_allow_html=True)

# ========================
# 📦 KONFIGURASI MODEL ENSEMBLE
# ========================
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# --- FILE ID DARI GOOGLE DRIVE PUBLIK ANDA ---
# PASTIKAN INI ADALAH FILE_ID YANG BENAR UNTUK MODEL ANDA!
GDRIVE_FILE_IDS = {
    'resnet': '1WZlTN5EMBTA06NMGNNYN6MyzeFQZPxPg', 
    'vgg': '11dXYwFYnhJB-t3JQhyETQxVvxrumaKwe',      
    'inception': '1STxuKijMG710wsPkFZhlqBVFgZbmO0Et' 
}

# Nama file lokal untuk model setelah diunduh di lingkungan Streamlit Cloud
MODEL_LOCAL_PATHS = {
    'resnet': 'resnet50_corn_leaf_disease.h5',
    'vgg': 'vgg16_corn_leaf_disease.h5',
    'inception': 'inceptionv3_corn_leaf_disease.h5'
}

@st.cache_resource(show_spinner=False, hash_funcs={"_thread.RLock": lambda _: None})
def load_all_models():
    """
    Memuat semua model Keras yang sudah terlatih.
    Jika model belum ada secara lokal (saat deployment awal di Streamlit Cloud),
    akan diunduh dari Google Drive menggunakan gdown.
    """
    loaded_models = {}
    for model_name, file_id in GDRIVE_FILE_IDS.items():
        local_path = MODEL_LOCAL_PATHS[model_name]
        
        # Buat direktori 'models' jika belum ada
        model_dir = os.path.dirname(local_path)
        if not os.path.exists(model_dir) and model_dir:
            os.makedirs(model_dir, exist_ok=True)

        if not os.path.exists(local_path):
            print(f"Mengunduh model {model_name} dari Google Drive...")
            try:
                gdown.download(id=file_id, output=local_path, quiet=True, fuzzy=True) 
                print(f"Model {model_name} berhasil diunduh!")
            except Exception as e:
                st.error(f"❌ ERROR: Gagal mengunduh model {model_name} dari Google Drive.")
                st.error(f"Pastikan FILE_ID ({file_id}) benar dan akses sharing link sudah diatur 'Anyone with the link'.")
                st.error(f"Detail error: {e}")
                st.stop()
        else:
            print(f"Model {model_name} sudah ada secara lokal. Melewatkan pengunduhan.")

        try:
            loaded_models[model_name] = load_model(local_path)
        except Exception as e:
            st.error(f"❌ ERROR: Gagal memuat model {model_name} dari file lokal ({local_path}).")
            st.error(f"Detail error: {e}")
            st.stop()

    return loaded_models['resnet'], loaded_models['vgg'], loaded_models['inception']

# Panggil fungsi untuk memuat model saat aplikasi Streamlit dimulai
resnet_model, vgg_model, inception_model = load_all_models()

def preprocess_image(image):
    """
    Melakukan preprocessing pada gambar untuk input model.
    Menerima objek PIL Image.
    """
    img_resized = image.resize(IMAGE_SIZE)
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = img_array / 255.0 
    return img_array

def ensemble_predict_streamlit(image_array):
    """
    Menggabungkan prediksi dari ResNet50, VGG16, dan InceptionV3
    menggunakan soft voting (rata-rata probabilitas).
    """
    pred_resnet = resnet_model.predict(image_array, verbose=0)
    pred_vgg = vgg_model.predict(image_array, verbose=0)
    pred_inception = inception_model.predict(image_array, verbose=0)

    ensemble_pred = (pred_resnet + pred_vgg + pred_inception) / 3.0

    predicted_class_index = np.argmax(ensemble_pred)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = ensemble_pred[0][predicted_class_index] * 100

    return predicted_class_name, confidence

# ========================
# 🤖 GEMINI API HANDLER
# ========================
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def get_gemini_api_key():
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
        if api_key:
            return api_key
    except Exception:
        pass

    local_key_path = os.path.join(os.path.dirname(__file__), "gemini_api_key.txt")
    if os.path.exists(local_key_path):
        with open(local_key_path, "r") as f:
            return f.read().strip()

    st.error("❌ **Error:** Kunci API Gemini tidak ditemukan.")
    st.markdown("""
        Untuk mendapatkan saran dari Gemini AI, Anda perlu mengatur kunci API Gemini Anda.
        * **Untuk deployment di Streamlit Cloud:** Tambahkan `GEMINI_API_KEY="YOUR_API_KEY_HERE"` di `.streamlit/secrets.toml` Anda.
        * **Untuk pengujian lokal:** Buat file `gemini_api_key.txt` di folder yang sama dengan `app.py` dan tempelkan kunci API Anda di dalamnya.
        Anda bisa mendapatkan API Key Gemini di [Google AI Studio](https://aistudio.google.com/app/apikey).
    """)
    return None

def clean_description(text):
    text = re.sub(r"<\/?\w+>", "", text) 
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text) 
    text = re.sub(r"^- ", "", text, flags=re.MULTILINE) 
    return text.strip()

def get_disease_description(label):
    api_key = get_gemini_api_key()
    if not api_key:
        return "Deskripsi penyakit tidak dapat dimuat karena kunci API Gemini tidak ditemukan."

    prompt = f"Jelaskan secara singkat tentang penyakit daun jagung {label} dalam 2-3 kalimat. Jawaban dalam teks biasa saja, tanpa tag HTML atau format markdown apa pun."
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={api_key}",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        response.raise_for_status()
        result = response.json()
        if "candidates" in result and len(result["candidates"]) > 0:
            raw_text = result["candidates"][0]["content"]["parts"][0]["text"]
            return clean_description(raw_text)
        else:
            return "Tidak ada deskripsi yang dihasilkan untuk penyakit ini."
    except requests.exceptions.RequestException as e:
        return f"❌ Gagal memuat deskripsi penyakit dari Gemini API (HTTP Error): {e}"
    except Exception as e:
        return f"❌ Gagal memuat deskripsi penyakit dari Gemini API: {e}"

def get_treatment_suggestions(disease_name):
    api_key = get_gemini_api_key()
    if not api_key:
        return "❌ Saran penanganan tidak dapat dimuat karena kunci API Gemini tidak ditemukan."

    prompt = f"Berikan saran penanganan dan saran tindakan pencegahan untuk menangani penyakit daun jagung: {disease_name}. Buat dalam bentuk poin yang masing-masing poin hanya berisi satu kalimat perintah singkat. Kelompokkan dalam dua bagian berbeda: 'Saran Penanganan:' dan 'Saran Pencegahan:'. Pastikan tidak ada format bold/italic atau tag HTML/Markdown lainnya. Gunakan tanda hubung (-) sebagai poin."
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={api_key}",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        response.raise_for_status()
        result = response.json()
        if "candidates" in result and len(result["candidates"]) > 0:
            raw_text = result["candidates"][0]["content"]["parts"][0]["text"]
            cleaned_text = re.sub(r"^\* ", "- ", raw_text, flags=re.MULTILINE)
            cleaned_text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", cleaned_text)
            cleaned_text = re.sub(r"(\*|_)(.*?)\1", r"\2", cleaned_text)
            
            cleaned_text = cleaned_text.replace("Saran Penanganan:", "\nSaran Penanganan:")
            cleaned_text = cleaned_text.replace("Saran Pencegahan:", "\nSaran Pencegahan:")

            return cleaned_text.strip()
        else:
            return "Tidak ada saran penanganan yang dihasilkan untuk penyakit ini."
    except requests.exceptions.RequestException as e:
        return f"❌ Gagal memanggil Gemini API untuk saran penanganan (HTTP Error): {e}"
    except Exception as e:
        return f"❌ Gagal memanggil Gemini API untuk saran penanganan: {e}"

# ========================
# 🎨 UI COMPONENTS
# ========================
def render_header():
    st.markdown(f"""
    <div style='background-color:#FFFFFF;
                box-shadow: 0 0 6px rgba(0,0,0,0.05);
                padding: 16px;
                border-radius: 10px;
                margin-bottom: 20px;'>
        <h1 style='color: {TITLE_COLOR};'>{APP_ICON} {APP_TITLE}</h1>
        <p style='font-size: 16px; color: {TEXT_COLOR};'>
            Unggah gambar daun jagung dan ketahui kondisi kesehatan tanaman jagung anda!
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_prediction_result(label, confidence, description):
    confidence = min(max(confidence, 0), 100)
    percent = round(confidence, 2)
    stroke_value = 440 * (percent / 100)

    st.markdown(
        f"""
        <style>
        .outer-box {{
            width: 100%;
            padding: 24px;
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 0 6px rgba(0,0,0,0.05);
            display: flex;
            flex-direction: row;
            gap: 24px;
            margin-bottom: 16px;
            flex-wrap: wrap;
            align-items: center;
            color: {TEXT_COLOR};
        }}

        .inner-box {{
            width: 100%;
            display: flex;
            flex-direction: row;
            gap: 24px;
            margin-bottom: 16px;
        }}

        .circle-container {{
            flex: 0 0 200px;
            max-width: 220px;
            aspect-ratio: 1;
            position: relative;
        }}

        svg {{
            width: 100%;
            height: 100%;
            transform: rotate(-90deg);
        }}

        .bg {{
            stroke: #eee;
            stroke-width: 10;
        }}

        .progress {{
            stroke: #89A827;
            stroke-linecap: round;
            stroke-dasharray: 440;
            stroke-dashoffset: 440;
            stroke-width: 10;
            animation: progressAnim 1s ease-out forwards;
        }}

        .circle-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            font-family: Arial, sans-serif;
        }}

        .circle-text .label {{
            font-size: 1.1em;
            font-weight: bold;
            color: #333;
        }}

        .circle-text .confidence {{
            font-size: 0.8em;
            color: #888;
        }}

        .description-box {{
            flex: 1;
            font-size: 1em;
            color: #444;
        }}

        @keyframes progressAnim {{
            to {{
                stroke-dashoffset: {440 - stroke_value};
            }}
        }}
        </style>
        <div class="outer-box">
            <div style="font-weight: 600; font-size: 20px; margin-bottom: 6px;">
                Seperti apa kondisi kesehatan tanaman jagung Anda?
            </div>
            <div class="inner-box">
                <div class="circle-container">
                    <svg viewBox="0 0 160 160" preserveAspectRatio="xMidYMid meet">
                        <circle class="bg" cx="80" cy="80" r="70" fill="none" />
                        <circle class="progress" cx="80" cy="80" r="70" fill="none" />
                    </svg>
                    <div class="circle-text">
                        <div class="label">{label}</div>
                        <div class="confidence">{percent:.2f}% yakin</div>
                    </div>
                </div>
                <div class="description-box">
                    <div>{description}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_treatment_suggestion(label, suggestion):
    st.markdown(
        f"""
        <div style="background-color:#FFFFFF;
                    padding:16px;
                    border-radius:10px;
                    box-shadow: 0 0 6px rgba(0,0,0,0.05);
                    padding: 16px;">
            <div style='font-weight: 600; font-size: 20px; margin-bottom: 16px; color: {TEXT_COLOR};'>Apa yang harus anda dilakukan?</div>
            <div style='font-size: 16px; color: {TEXT_COLOR};'>{suggestion}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_custom_spinner(message="🔄 Memproses...", color="#89A827", text_color="#444"):
    return f"""
    <div style="display:flex; align-items:center; gap:10px; padding:12px 0;">
        <div style="
            border: 6px solid #eee;
            border-top: 6px solid {color};
            border-radius: 50%;
            width: 26px;
            height: 26px;
            animation: spin 1s linear infinite;">
        </div>
        <div style="color:{text_color}; font-size:16px;">
            {message}
        </div>
    </div>
    <style>
    @keyframes spin {{
      0% {{ transform: rotate(0deg); }}
      100% {{ transform: rotate(360deg); }}
    }}
    </style>
    """
def render_footer():
    st.markdown('<div style="height: 10vh;"></div>', unsafe_allow_html=True) 
    st.markdown(
        """
        <hr style="border: none; height: 2px; background-color: #89A827; margin-top: 60px;">
        <footer style="text-align: center; color: #555; font-size: 0.9em; padding: 12px;">
            Aplikasi ini menggunakan model <strong>Ensemble CNN</strong> dan <strong>Google Gemini</strong>.<br>
            © 2025 | Dibuat dengan ❤️ oleh Tim Honda Mio<br>
            <a href="mailto:oscariqbal75@gmail.com" style="color: #89A827; text-decoration: none;">Hubungi Kami</a>
        </footer>
        """,
        unsafe_allow_html=True
    )

# ========================
# 🚀 MAIN APP LOGIC
# ========================
def main():
    apply_custom_styles()
    render_header()

    # Inisialisasi variabel state sesi
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
    if "show_result" not in st.session_state:
        st.session_state.show_result = False
    if "upload_key" not in st.session_state: 
        st.session_state.upload_key = "uploader_1"

    # Fungsi untuk mereset prediksi
    def reset_prediction():
        st.session_state.uploaded_file = None
        st.session_state.show_result = False
        st.session_state.upload_key = f"uploader_{np.random.randint(10000)}" 
        st.rerun() 

    # Buat dua kolom untuk tata letak
    col1, col2 = st.columns([1.5, 1]) 

    # Logika untuk menampilkan pengunggah file atau hasil prediksi
    if st.session_state.uploaded_file is None:
        with col1:
            # Hapus argumen use_container_width dari st.file_uploader
            uploaded = st.file_uploader("Unggah gambar daun jagung...", type=["jpg", "jpeg", "png"], key=st.session_state.upload_key)
            
            if uploaded is not None and uploaded != st.session_state.uploaded_file:
                st.session_state.uploaded_file = uploaded
                st.session_state.show_result = True
                st.rerun() 
            elif uploaded is None and st.session_state.uploaded_file is not None:
                reset_prediction() 
        with col2:
            st.empty() 
    else: 
        uploaded_file = st.session_state.uploaded_file

        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error("❌ Gagal membuka gambar. Pastikan ini adalah file gambar yang valid.")
            st.exception(e)
            reset_prediction() 
            return

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        with col1: # Kolom kiri hanya untuk gambar
            st.markdown(
                f"""<div class="image-display-box" style="text-align:center;">
                    <img src="data:image/png;base64,{img_str}" 
                        style="width:100%; max-width: 600px; height: auto; 
                                background-color:#FFFFFF; margin-bottom: 10px; padding:16px; 
                                border-radius:10px; box-shadow: 0 0 6px rgba(0,0,0,0.05);"/>
                </div>""",
                unsafe_allow_html=True
            )

        with col2: # Kolom kanan untuk prediksi dan saran
            # Jalankan prediksi model
            spinner_model = st.empty()
            spinner_model.markdown(render_custom_spinner("🔍 Menganalisis gambar dengan Ensemble Model..."), unsafe_allow_html=True)
            processed_image = preprocess_image(image)
            try:
                label, confidence = ensemble_predict_streamlit(processed_image)
            except Exception as e:
                spinner_model.empty()
                st.error("❌ Terjadi kesalahan saat prediksi model ensemble. Pastikan model dimuat dengan benar.")
                st.exception(e)
                reset_prediction()
                return
            spinner_model.empty() 
            
            # Tampilkan hasil prediksi
            description = get_disease_description(label)
            render_prediction_result(label, confidence, description)

            st.markdown("<br>", unsafe_allow_html=True)

            # Tampilkan saran penanganan dari Gemini
            spinner_gemini = st.empty()
            spinner_gemini.markdown(render_custom_spinner("🧠 Mengambil saran penanganan dari Gemini..."), unsafe_allow_html=True)
            suggestion = get_treatment_suggestions(label)
            spinner_gemini.empty() 
            render_treatment_suggestion(label, suggestion)
            
            st.button("🔁 Prediksi Lagi!", use_container_width=True, on_click=reset_prediction)


    render_footer() 

if __name__ == "__main__":
    main()