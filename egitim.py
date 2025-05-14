import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import pandas as pd
import pickle
import json

# Veri klasörlerinin yolları
TRAIN_DIR = 'content/ieee-mbl-cls/train'
VAL_DIR = 'content/ieee-mbl-cls/val'

# Model parametreleri
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 4  # Unripe, Ripe, Old, Damaged

def olustur_model():
    """CNN modelini oluşturur"""
    model = Sequential([
        # Konvolüsyon katmanları
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        
        # Düzleştirme ve tam bağlantılı katmanlar
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Modeli derleme
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def egit_model():
    # Veri artırma için generator'lar
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
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Veri akışı oluşturma
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    # Sınıf isimlerini kaydet
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    
    # Model oluştur
    model = olustur_model()
    
    # Callback'ler
    checkpoint = ModelCheckpoint(
        'en_iyi_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )
    
    # Eğitim
    print("Model eğitimi başlıyor...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping]
    )
    
    # Model ve sınıf isimlerini kaydet
    model.save('meyve_siniflandirma_modeli.h5')
    
    # Eğitim geçmişini kaydet
    with open('egitim_history.pickle', 'wb') as file:
        pickle.dump(history.history, file)
    print("Eğitim geçmişi 'egitim_history.pickle' olarak kaydedildi.")
    
    # Sınıf isimlerini kaydet
    with open('sinif_isimleri.json', 'w') as f:
        json.dump(class_names, f)
    
    print("Eğitim tamamlandı. Model kaydedildi.")
    
    # Confusion Matrix ve F1 skorları için veri hazırlığı
    print("Confusion Matrix ve F1 skorları hesaplanıyor...")
    
    # Validation veri setinden tahminler yapma
    val_generator_cm = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=1,  # Tek tek tahmin etmek için
        class_mode='categorical',
        shuffle=False  # Sıralamayı korumak için
    )
    
    # Tüm validation verileri üzerinden tahmin yap
    y_true = []
    y_pred = []
    val_generator_cm.reset()
    
    steps = val_generator_cm.samples
    for i in range(steps):
        x, y = val_generator_cm.next()
        y_true.append(np.argmax(y))
        
        pred = model.predict(x, verbose=0)
        y_pred.append(np.argmax(pred))
    
    # Sınıf isimlerini alalım (Türkçe)
    class_names_tr = []
    for i in range(NUM_CLASSES):
        original_name = class_names.get(str(i), f"Sınıf {i}")
        tr_name = {
            "Unripe": "Olgunlaşmamış",
            "Ripe": "Olgun", 
            "Old": "Yaşlı",
            "Damaged": "Hasarlı"
        }.get(original_name, original_name)
        class_names_tr.append(tr_name)
    
    return model, history, y_true, y_pred, class_names_tr

def gorselleştir_confusion_matrix(y_true, y_pred, class_names):
    """Confusion Matrix'i görselleştirir"""
    # Confusion matrix hesapla
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix 'confusion_matrix.png' olarak kaydedildi.")

def gorselleştir_f1_skorlari(y_true, y_pred, class_names):
    """F1 skorlarını görselleştirir"""
    # Sınıflandırma raporu
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # F1 skorlarını al
    f1_scores = {}
    for sınıf, metrics in report.items():
        if sınıf in class_names:
            f1_scores[sınıf] = metrics['f1-score']
    
    # F1 skorlarını görselleştir
    plt.figure(figsize=(10, 6))
    plt.bar(f1_scores.keys(), f1_scores.values())
    plt.title('F1 Skorları')
    plt.xlabel('Sınıflar')
    plt.ylabel('F1 Skoru')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('f1_skorlari.png')
    print("F1 skorları 'f1_skorlari.png' olarak kaydedildi.")
    
    # Genel F1 skoru
    genel_f1 = report['macro avg']['f1-score']
    print(f"Genel F1 Skoru (Macro): {genel_f1:.4f}")
    
    # Detaylı F1 skorları
    print("\nF1 Skorları (Sınıf bazında):")
    for sınıf, f1 in f1_scores.items():
        print(f"{sınıf}: {f1:.4f}")

if __name__ == "__main__":
    # GPU kullanımını kontrol et
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"{len(gpus)} GPU bulundu, kullanıma hazırlanıyor...")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("GPU bulunamadı, CPU kullanılacak.")
    
    # Modeli eğit
    model, history, y_true, y_pred, class_names_tr = egit_model()
    
    # Confusion Matrix ve F1 skorlarını görselleştir
    gorselleştir_confusion_matrix(y_true, y_pred, class_names_tr)
    gorselleştir_f1_skorlari(y_true, y_pred, class_names_tr)
    
    # Eğitim sonuçlarını görselleştir
    plt.figure(figsize=(12, 10))
    
    # Doğruluk grafiği
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Doğruluğu')
    plt.ylabel('Doğruluk')
    plt.xlabel('Epoch')
    plt.legend(['Eğitim', 'Doğrulama'], loc='lower right')
    
    # Kayıp grafiği
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Kaybı')
    plt.ylabel('Kayıp')
    plt.xlabel('Epoch')
    plt.legend(['Eğitim', 'Doğrulama'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('egitim_grafikleri.png')
    plt.show() 