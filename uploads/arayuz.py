import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gradio as gr
import traceback

# Model dosyası ve sınıf isimleri
MODEL_PATH = 'C:\\Users\\hasan\\OneDrive\\Masaüstü\\bulutproje\\en_iyi_model.h5'
CLASS_NAMES_PATH = 'C:\\Users\\hasan\\OneDrive\\Masaüstü\\bulutproje\\sinif_isimleri.json'

# Görüntü boyutları
IMG_WIDTH, IMG_HEIGHT = 224, 224

def yuklenen_modeli_kontrol_et():
    """Model ve sınıf isimleri dosyalarının varlığını kontrol eder"""
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
            print("Model veya sınıf isimleri dosyası eksik!")
            return False, None, None
        
        model = load_model(MODEL_PATH, compile=False)
        print(f"Model yüklendi: {MODEL_PATH}")
        
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        print(f"Sınıf isimleri yüklendi: {class_names}")
        
        return True, model, class_names
    except Exception as e:
        print(f"Model yükleme hatası: {str(e)}")
        print(traceback.format_exc())
        return False, None, None

def goruntu_siniflandir(img):
    """Yüklenen görüntüyü sınıflandırır"""
    if img is None:
        print("Geçersiz görüntü.")
        return {}

    try:
        success, model, class_names = yuklenen_modeli_kontrol_et()
        if not success:
            print("Model yüklenemedi.")
            return {}
        
        img_array = image.img_to_array(img)
        img_array = tf.image.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        
        results = {}
        for i, prob in enumerate(predictions[0]):
            class_name = class_names.get(str(i), f"Sınıf {i}")
            tr_name = {
                "Unripe": "Olgunlaşmamış",
                "Ripe": "Olgun",
                "Old": "Yaşlı",
                "Damaged": "Hasarlı"
            }.get(class_name, class_name)
            results[tr_name] = float(prob)
        
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        return sorted_results

    except Exception as e:
        print(f"Tahmin hatası: {str(e)}")
        print(traceback.format_exc())
        return {}

def main():
    title = "Meyve Olgunluk Sınıflandırma Sistemi"
    description = """
    Bu uygulama, meyveleri dört kategoriden birine sınıflandırır:
    - Olgunlaşmamış (Unripe)
    - Olgun (Ripe)
    - Yaşlı (Old)
    - Hasarlı (Damaged)
    
    Lütfen bir meyve görüntüsü yükleyin veya örnek görüntülerden birini seçin.
    """
    
    examples = []  # Örnek görseller tanımlanabilir

    try:
        demo = gr.Interface(
            fn=goruntu_siniflandir,
            inputs=gr.Image(type="pil"),
            outputs=gr.Label(num_top_classes=4),
            title=title,
            description=description,
            examples=examples,
            theme="default"
        )
        demo.launch(share=True)
    except Exception as e:
        print(f"Arayüz hatası: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
