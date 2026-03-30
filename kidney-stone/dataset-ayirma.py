import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. PARAMETRELER (Puan: 7, 10)
BASE_DIR = '/Users/flawi/Desktop/archive' # 'Normal' ve 'Stone' klasörlerinin olduğu ana dizin
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 2. VERİ AYRIMI STRATEJİSİ (Puan: 7)
# Test seti için %10 ayırıyoruz (validation_split burada ana ayrımı yapar)
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

test_generator = test_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation', # %10 Test kısmı
    shuffle=False
)

# Geri kalan %90'ı Eğitim (%70) ve Doğrulama (%20) olarak bölmek için:
# Toplamın %20'si, %90'lık kısmın yaklaşık %22'sine denk gelir.
train_val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.22, # %90 içinden %22 ayırınca toplamın %20'si Val olur
    rotation_range=20,     # Augmentasyon (Puan: 6)
    horizontal_flip=True
)

train_generator = train_val_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',    # %70 Eğitim
    shuffle=True
)

val_generator = train_val_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',  # %20 Doğrulama
    shuffle=True
)

print(f"Eğitim örneği: {train_generator.samples}")
print(f"Doğrulama örneği: {val_generator.samples}")
print(f"Test örneği: {test_generator.samples}")