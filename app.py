import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import gdown 

# --- 1. Konfigurasi Global ---
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# --- FILE ID DARI GOOGLE DRIVE PUBLIK ---
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

# --- 2. Muat Model (Sekali Saat Aplikasi Dimulai) ---
@st.cache_resource # Dekorator ini menyimpan model dalam cache agar tidak dimuat ulang setiap interaksi pengguna
def load_all_models():
    """
    Memuat semua model Keras yang sudah terlatih.
    Jika model belum ada secara lokal (saat deployment awal di Streamlit Cloud),
    akan diunduh dari Google Drive menggunakan gdown.
    """
    loaded_models = {}
    for model_name, file_id in GDRIVE_FILE_IDS.items():
        local_path = MODEL_LOCAL_PATHS[model_name]

        # Cek apakah file model sudah ada secara lokal
        if not os.path.exists(local_path):
            st.info(f"Mengunduh model {model_name} dari Google Drive... Ini mungkin memakan waktu.")
            try:
                # Mengunduh menggunakan gdown
                gdown.download(id=file_id, output=local_path, quiet=False)
                st.success(f"Model {model_name} berhasil diunduh!")
            except Exception as e:
                st.error(f"ERROR: Gagal mengunduh model {model_name} dari Google Drive.")
                st.error(f"Pastikan FILE_ID ({file_id}) benar dan akses sharing link sudah diatur 'Anyone with the link'.")
                st.error(f"Detail error: {e}")
                st.stop() # Hentikan aplikasi jika gagal mengunduh
        else:
            st.info(f"Model {model_name} sudah ada secara lokal. Melewatkan pengunduhan.")

        try:
            # Muat model dari path lokal
            loaded_models[model_name] = load_model(local_path)
        except Exception as e:
            st.error(f"ERROR: Gagal memuat model {model_name} dari file lokal ({local_path}).")
            st.error(f"Detail error: {e}")
            st.stop() # Hentikan aplikasi jika gagal memuat model

    return loaded_models['resnet'], loaded_models['vgg'], loaded_models['inception']

# Panggil fungsi untuk memuat model saat aplikasi Streamlit dimulai
resnet_model, vgg_model, inception_model = load_all_models()

# --- 3. Fungsi Preprocessing Gambar ---
def preprocess_image(image):
    """
    Melakukan preprocessing pada gambar untuk input model.
    Menerima objek PIL Image.
    """
    img_resized = image.resize(IMAGE_SIZE)
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch
    img_array = img_array / 255.0 # Normalisasi piksel
    return img_array

# --- 4. Fungsi Prediksi Ensemble (Soft Voting) ---
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

# --- 5. Antarmuka Pengguna Streamlit ---
st.set_page_config(
    page_title="Deteksi Penyakit Daun Jagung",
    page_icon="🌽",
    layout="centered"
)

st.title("🌽 Deteksi Penyakit Daun Jagung dengan Ensemble CNN")
st.write("Unggah gambar daun jagung untuk mendeteksi jenis penyakitnya (Blight, Common Rust, Gray Leaf Spot, atau Healthy).")

uploaded_file = st.file_uploader("Pilih gambar daun jagung...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang Diunggah', use_column_width=True)
    st.write("")
    st.write("Menganalisis gambar...")

    processed_image = preprocess_image(image)
    predicted_class, confidence = ensemble_predict_streamlit(processed_image)

    st.success(f"**Hasil Prediksi:** {predicted_class}")
    st.info(f"**Tingkat Kepercayaan:** {confidence:.2f}%")

    st.markdown(
        """
        ---
        **Catatan:**
        * Model ini dilatih untuk mendeteksi 'Blight', 'Common_Rust', 'Gray_Leaf_Spot', dan 'Healthy'.
        * Keakuratan prediksi dapat bervariasi tergantung kualitas gambar dan kompleksitas penyakit.
        """
    )