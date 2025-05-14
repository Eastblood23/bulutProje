import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import pandas as pd
import json
import pickle

# Sabit değerler
VAL_DIR = 'content/ieee-mbl-cls/val'
IMG_WIDTH, IMG_HEIGHT = 224, 224
MODEL_PATH = 'en_iyi_model.h5'

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
    plt.show()
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
    plt.show()
    print("F1 skorları 'f1_skorlari.png' olarak kaydedildi.")
    
    # Genel F1 skoru
    genel_f1 = report['macro avg']['f1-score']
    print(f"Genel F1 Skoru (Macro): {genel_f1:.4f}")
    
    # Detaylı F1 skorları
    print("\nF1 Skorları (Sınıf bazında):")
    for sınıf, f1 in f1_scores.items():
        print(f"{sınıf}: {f1:.4f}")
    
    return report

def gorselleştir_egitim_grafikleri(model):
    """Modelin eğitim geçmişini görselleştirir (accuracy ve loss grafikleri)"""
    # Önce kaydedilmiş history dosyasını kontrol et
    if os.path.exists('egitim_history.pickle'):
        try:
            with open('egitim_history.pickle', 'rb') as file:
                history = pickle.load(file)
                print("Eğitim geçmişi 'egitim_history.pickle' dosyasından yüklendi.")
        except Exception as e:
            print(f"History dosyası yüklenirken hata: {e}")
            history = None
    else:
        # Modelden history almayı dene
        try:
            history = model.history.history
            if not history:
                print("Model history bilgisi boş.")
                return
        except:
            print("Model eğitim geçmişi bulunamadı. Kaydedilmiş bir history dosyası da yok.")
            return
    
    if not history:
        print("Eğitim geçmişi bulunamadı.")
        return
    
    # Grafikleri çiz
    plt.figure(figsize=(12, 10))
    
    # Doğruluk grafiği
    plt.subplot(2, 1, 1)
    if 'accuracy' in history:
        plt.plot(history['accuracy'], label='Eğitim')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Doğrulama')
    plt.title('Model Doğruluğu')
    plt.ylabel('Doğruluk')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Kayıp grafiği
    plt.subplot(2, 1, 2)
    if 'loss' in history:
        plt.plot(history['loss'], label='Eğitim')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Doğrulama')
    plt.title('Model Kaybı')
    plt.ylabel('Kayıp')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('egitim_grafikleri.png')
    plt.show()
    print("Eğitim grafikleri 'egitim_grafikleri.png' olarak kaydedildi.")

def main():
    # GPU kullanımını kontrol et
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"{len(gpus)} GPU bulundu, kullanıma hazırlanıyor...")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("GPU bulunamadı, CPU kullanılacak.")
    
    # Modeli yükle
    print(f"Model yükleniyor: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    
    # Sınıf isimlerini yükle
    try:
        with open('sinif_isimleri.json', 'r') as f:
            class_names = json.load(f)
    except:
        print("sinif_isimleri.json bulunamadı. Varsayılan sınıf isimleri kullanılacak.")
        class_names = {
            "0": "Unripe",
            "1": "Ripe",
            "2": "Old",
            "3": "Damaged"
        }
    
    # Sınıfların Türkçe karşılıklarını hazırla
    class_names_tr = []
    for i in range(len(class_names)):
        original_name = class_names.get(str(i), f"Sınıf {i}")
        tr_name = {
            "Unripe": "Olgunlaşmamış",
            "Ripe": "Olgun", 
            "Old": "Yaşlı",
            "Damaged": "Hasarlı"
        }.get(original_name, original_name)
        class_names_tr.append(tr_name)
    
    # Eğitim grafiklerini görselleştirmeyi dene
    print("\nEğitim grafiklerini oluşturmaya çalışılıyor...")
    gorselleştir_egitim_grafikleri(model)
    
    # Validation veri seti için datagen
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    print(f"Validation verileri yükleniyor: {VAL_DIR}")
    # Validation veri setinden tahminler yapma
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=1,  # Tek tek tahmin etmek için
        class_mode='categorical',
        shuffle=False  # Sıralamayı korumak için
    )
    
    # Tüm validation verileri üzerinden tahmin yap
    y_true = []
    y_pred = []
    val_generator.reset()
    
    steps = val_generator.samples
    print(f"Toplam {steps} adet validation verisi değerlendiriliyor...")
    
    for i in range(steps):
        if i % 20 == 0:
            print(f"İşlenen görüntü: {i}/{steps}")
        
        x, y = next(val_generator)
        y_true.append(np.argmax(y))
        
        pred = model.predict(x, verbose=0)
        y_pred.append(np.argmax(pred))
    
    print("Değerlendirme tamamlandı.")
    
    # Confusion Matrix ve F1 skorlarını görselleştir
    print("\nConfusion Matrix oluşturuluyor...")
    gorselleştir_confusion_matrix(y_true, y_pred, class_names_tr)
    
    print("\nF1 Skorları hesaplanıyor...")
    gorselleştir_f1_skorlari(y_true, y_pred, class_names_tr)

if __name__ == "__main__":
    main() 