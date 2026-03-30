import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import os
import random  # Rastgele seçim için eklendi
import gdown   # Drive'dan model indirmek için eklendi

# --- 1. SAYFA AYARLARI---
st.set_page_config(page_title="Böbrek Taşı Teşhis Sistemi", page_icon="🔬", layout="wide")

# Bilgisayarındaki ana klasör yolu (GitHub'da ./ kullanımı uygundur)
BASE_PATH = './' 

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stTable { background-color: #ffffff; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODEL YÜKLEME (Bulut Entegrasyonu) ---
@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_PATH, 'kidney_stone_cnn_model.h5')
    
    # Eğer model yerelde yoksa Google Drive'dan indir
    if not os.path.exists(model_path):
        with st.spinner("Model dosyası büyük olduğu için buluttan indiriliyor, lütfen bekleyin..."):
            file_id = '1Jh5yxt587354eAs6Zz6ZQlLY79zg3mH7' 
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(url, model_path, quiet=False)
            except Exception as e:
                st.error(f"Model indirilirken hata oluştu: {e}")
                return None
                
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_model()

# --- 3. YAN MENÜ ---
with st.sidebar:
    st.title("👤 Öğrenci Bilgileri")
    st.info("""
    **Ad Soyad:** Habib Can Değirmenci 
    **Öğrenci No:** 220706025  
    **Proje:** Böbrek Taşı Sınıflandırma
    """)
    
    st.divider()
    sayfa = st.radio("Menü Gezinme", ["🏠 Ana Sayfa", "📊 Model Analizi", "📄 Proje Hakkında"])

# --- SAYFA 1: ANA SAYFA & TAHMİN ---
if sayfa == "🏠 Ana Sayfa":
    st.title("🔬 Böbrek Taşı Otomatik Teşhis Sistemi")
    st.write("Sistem, yüklenen radyolojik görüntüleri analiz ederek taş varlığını tespit eder.")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📸 Görüntü Yükleme")
        file = st.file_uploader("Analiz için görüntü seçin...", type=["jpg", "jpeg", "png"])
        if file:
            img = Image.open(file)
            st.image(img, use_container_width=True, caption="İşlenecek Görüntü")
            
    with col2:
        st.subheader("🩺 Analiz Sonucu")
        if file and model:
            with st.spinner("Yapay zeka analiz ediyor..."):
                img_proc = img.convert("RGB").resize((224, 224))
                img_array = np.array(img_proc) / 255.0
                prediction = model.predict(np.expand_dims(img_array, axis=0))
                
                res_idx = np.argmax(prediction)
                prob = np.max(prediction) * 100
                
                if res_idx == 1:
                    st.error(f"### SONUÇ: TAŞ TESPİT EDİLDİ")
                    st.metric("Güven Skoru", f"%{prob:.2f}")
                else:
                    st.success(f"### SONUÇ: NORMAL (SAĞLIKLI)")
                    st.metric("Güven Skoru", f"%{prob:.2f}")
        else:
            st.info("Sonuç için lütfen bir görüntü yükleyin.")

# --- SAYFA 2: MODEL ANALİZİ ---
elif sayfa == "📊 Model Analizi":
    st.title("📈 Başarım ve Grafik Analizleri")
    
    st.subheader("📋 Sınıflandırma Raporu (Test Verisi)")
    report_df = pd.DataFrame({
        "Sınıf": ["Normal", "Stone"],
        "Precision": [0.94, 1.00],
        "Recall": [1.00, 0.95],
        "F1-Score": [0.97, 0.97]
    })
    st.table(report_df)
    
    st.divider()
    
    st.subheader("📉 Eğitim Süreci")
    c1, c2 = st.columns(2)
    with c1:
        st.image(os.path.join(BASE_PATH, 'accuracy_plot.png'), caption="Accuracy Grafiği")
    with c2:
        st.image(os.path.join(BASE_PATH, 'loss_plot.png'), caption="Loss Grafiği")
        
    st.divider()
    
    st.subheader("🎯 Hata Dağılımı ve ROC Eğrisi")
    c3, c4 = st.columns(2)
    with c3:
        st.image(os.path.join(BASE_PATH, 'confusion_matrix.png'), caption="Confusion Matrix")
    with c4:
        st.image(os.path.join(BASE_PATH, 'roc_curve.png'), caption="ROC Eğrisi")

    # --- RASTGELE FOTOĞRAF BÖLÜMÜ (HATA DENETİMLİ) ---
    st.markdown("### 2.1. Veri Setinden Rastgele Örnekler")

    test_normal_dir = './proje_verileri/test/Normal'
    test_stone_dir = './proje_verileri/test/Stone'

    col_img1, col_img2 = st.columns(2)

    # Normal klasörü kontrolü
    if os.path.exists(test_normal_dir):
        normal_images = [f for f in os.listdir(test_normal_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if normal_images:
            random_normal = random.choice(normal_images)
            img_path = os.path.join(test_normal_dir, random_normal)
            col_img1.image(img_path, caption=f"Normal Örnek: {random_normal}", use_container_width=True)
        else:
            col_img1.warning(f"Klasör bulundu ama içinde resim yok: {test_normal_dir}")
    else:
        col_img1.error(f"KLASÖR BULUNAMADI: {test_normal_dir}")

    # Stone klasörü kontrolü
    if os.path.exists(test_stone_dir):
        stone_images = [f for f in os.listdir(test_stone_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if stone_images:
            random_stone = random.choice(stone_images)
            img_path = os.path.join(test_stone_dir, random_stone)
            col_img2.image(img_path, caption=f"Stone Örnek: {random_stone}", use_container_width=True)
        else:
            col_img2.warning(f"Klasör bulundu ama içinde resim yok: {test_stone_dir}")
    else:
        col_img2.error(f"KLASÖR BULUNAMADI: {test_stone_dir}")
    
    st.divider()
    
    st.markdown("### 3. Model Mimarisi")
    st.write("Model, 3 ardışık Convolutional katman, MaxPooling ve yoğunluk (Dense) katmanlarından oluşmaktadır.")
    
    st.markdown("### 4. Akademik Sonuç")
    st.success("Model, Stone sınıfında %100 Precision başarısı göstererek yanlış teşhis oranını sıfıra indirmiştir.")
    
# --- SAYFA 3: PROJE HAKKINDA  ---
elif sayfa == "📄 Proje Hakkında":
    st.title("📝 Proje Detayları ve Veri Metrikleri")
    
    st.markdown("### 1. Problem Tanımı")
    st.write("Böbrek taşlarının erken ve doğru teşhisi, tedavi sürecini hızlandırarak hastaların yaşam kalitesini artırır. Bu çalışma, radyologlara karar destek mekanizması sunmayı amaçlar.")
    
    st.markdown("### 2. Veri Seti ve Dağılımı")
    st.write("**Kaynak:** Kaggle - Kidney Stone | Classification and Object Detection - https://www.kaggle.com/datasets/imtkaggleteam/kidney-stone-classification-and-object-detection")
    
    # Veri Dağılım Tablosu
    data_split = {
        "Aşama (Set)": ["Eğitim (Train)", "Doğrulama (Val)", "Test (Hold-out)", "TOPLAM"],
        "Görüntü Sayısı": ["6.608", "1.888", "944", "9.440"],
        "Oran": ["%70", "%20", "%10", "%100"]
    }
    st.table(pd.DataFrame(data_split))
    
    st.markdown("""
    - **Sınıflar:** Normal ve Stone (Taşlı)
    - **Giriş Boyutu:** 224x224 piksel (RGB)
    - **Ön İşleme:** Rescaling (1./255) ve Veri Artırma (Augmentation) uygulanmıştır.
    """)
    
    st.markdown("### 3. Model Mimarisi")
    st.write("Model, 3 ardışık Convolutional (Evrişimli) katman, MaxPooling ve yoğunluk (Dense) katmanlarından oluşmaktadır. Aşırı öğrenmeyi engellemek adına Dropout (%50) kullanılmıştır.")
    
    st.markdown("### 4. Akademik Sonuç ve Değerlendirme")
    st.success("""
    Yapılan testler sonucunda model, Stone sınıfında **%100 Precision (Kesinlik)** değerine ulaşmıştır. 
    Bu, sistemin 'Taş Var' dediği hiçbir vakada yanılmadığını ve sağlıklı bireylere yanlış teşhis koymadığını göstermektedir. 
    Genel doğruluk (Accuracy) oranı **%97** olarak kaydedilmiştir.
    """)