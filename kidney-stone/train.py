import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import seaborn as sns

# --- 1. VERİ SETİ YOLLARI VE PARAMETRELER ---
TRAIN_DIR = '/Users/flawi/Desktop/proje_verileri/train'
VAL_DIR   = '/Users/flawi/Desktop/proje_verileri/val'
TEST_DIR  = '/Users/flawi/Desktop/proje_verileri/test'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25

# --- 2. VERİ ÖN İŞLEME VE AUGMENTATION ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
)

# --- 3. MODEL MİMARİSİ (CNN) ---
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# --- 4. MODEL DERLEME ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

# --- 5. EĞİTİM ---
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\n--- Eğitim Başlıyor ---")
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, callbacks=[early_stop])

# --- 6. MODELİ KAYDETME ---
model.save('kidney_stone_cnn_model.h5')

# --- 7. PERFORMANS GRAFİKLERİ ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Başarımı')
plt.plot(history.history['val_accuracy'], label='Doğrulama Başarımı')
plt.title('Model Doğruluk Grafiği')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Model Kayıp Grafiği')
plt.legend()
plt.show()

# --- 8. TAHMİNLERİN HESAPLANMASI (KRİTİK ADIM) ---
print('\n--- Test Verisi Tahminleri Yapılıyor ---')
# Önce tahminleri yapıyoruz ki grafik fonksiyonlarına veri gitsin
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# --- 9. KARIŞIKLIK MATRİSİ GÖRSELLEŞTİRME ---
def plot_confusion_matrix(test_gen, predictions):
    cm = confusion_matrix(test_gen.classes, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=test_gen.class_indices.keys(),
                yticklabels=test_gen.class_indices.keys())
    plt.title('Karışıklık Matrisi (Confusion Matrix)')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.savefig('confusion_matrix.png') 
    plt.show()

# --- 10. ROC EĞRİSİ GÖRSELLEŞTİRME ---
def plot_roc_curve(test_gen, probabilities):
    # İkili sınıflandırmada taş (stone) sınıfı olasılıklarını kullanıyoruz
    fpr, tpr, _ = roc_curve(test_gen.classes, probabilities[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (Alan = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Yanlış Pozitif Oranı (FPR)')
    plt.ylabel('Doğru Pozitif Oranı (TPR)')
    plt.title('ROC Eğrisi (Receiver Operating Characteristic)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve.png')
    plt.show()

# --- 11. SONUÇLARI ÇALIŞTIR VE RAPORLA ---
plot_confusion_matrix(test_generator, y_pred)
plot_roc_curve(test_generator, Y_pred)

print('\n--- Sınıflandırma Raporu ---')
target_names = list(test_generator.class_indices.keys())
print(classification_report(test_generator.classes, y_pred, target_names=target_names))