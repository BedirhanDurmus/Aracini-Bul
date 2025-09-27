# 🚗 Araç Fiyat Tahmin Uygulaması

Bu uygulama, makine öğrenmesi kullanarak araç fiyatlarını tahmin eden kullanıcı dostu bir web arayüzüdür.

## 🎯 Özellikler

- **Kullanıcı Dostu Arayüz**: Streamlit ile modern ve responsive tasarım
- **Kapsamlı Araç Bilgileri**: Marka, model, teknik özellikler ve durum bilgileri
- **Yüksek Doğruluk**: XGBoost modeli ile %90+ doğruluk oranı
- **Gerçek Zamanlı Tahmin**: Anında fiyat tahmini
- **Detaylı Bilgi Girişi**: 15+ farklı araç özelliği

## 📊 Desteklenen Özellikler

### Temel Bilgiler
- Marka, Seri, Model
- Yıl, Kilometre

### Teknik Özellikler
- Vites Tipi (Düz, Otomatik, Yarı Otomatik)
- Yakıt Türü (Benzin, Dizel, Hibrit, Elektrik, LPG & Benzin)
- Kasa Tipi (Sedan, Hatchback, Coupe, vb.)
- Renk
- Çekiş Tipi (Önden Çekiş, Arkadan İtiş, 4WD)

### Motor Bilgileri
- Motor Hacmi (cc)
- Motor Gücü (HP)

### Durum Bilgileri
- Araç Vergisi (TL)
- Tramer Tutarı (TL)
- Boya Parça Durumu

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.8+
- Gerekli kütüphaneler (requirements.txt)

### Kurulum Adımları

1. **Projeyi klonlayın:**
```bash
git clone <repository-url>
cd aracınıBulDataScience
```

2. **Gerekli kütüphaneleri yükleyin:**
```bash
pip install -r requirements.txt
```

3. **Uygulamayı çalıştırın:**
```bash
# Yöntem 1: Otomatik çalıştırıcı
python run_app.py

# Yöntem 2: Manuel çalıştırma
streamlit run app.py

# Yöntem 3: Python modülü olarak
python -m streamlit run app.py
```

4. **Tarayıcınızda açın:**
```
http://localhost:8501
```

### 🔍 Model Açıklanabilirliği
Ayrı bir açıklanabilirlik sayfası için:
```bash
streamlit run explainability_page.py
```

## 📁 Dosya Yapısı

```
aracınıBulDataScience/
├── app.py                      # Ana Streamlit uygulaması
├── main.ipynb                  # Veri analizi ve model eğitimi
├── cars_tr.csv                 # Araç veri seti
├── best_car_price_model.pkl    # Eğitilmiş XGBoost modeli
├── requirements.txt            # Python bağımlılıkları
└── README.md                   # Bu dosya
```

## 🔧 Model Detayları

- **Algoritma**: XGBoost Regressor
- **Veri Seti**: 6,675 araç kaydı
- **Özellik Sayısı**: 43 (encoded)
- **Doğruluk Oranı**: %90+
- **Cross-Validation**: 5-fold

### Veri Ön İşleme
- Label Encoding (Marka, Seri, Model, Kasa Tipi, Çekiş Tipi)
- One-Hot Encoding (Vites, Yakıt, Renk)
- Eksik değer doldurma
- Outlier temizleme

## 💡 Kullanım

1. **Sol menüden araç bilgilerinizi girin**
2. **Marka, seri ve model seçimlerini yapın**
3. **Teknik özellikleri ve durum bilgilerini doldurun**
4. **"Fiyat Tahmini Yap" butonuna tıklayın**
5. **Tahmin edilen fiyatı görün**

## 📈 Performans Metrikleri

- **MAE (Mean Absolute Error)**: ~77,000 TL
- **RMSE (Root Mean Square Error)**: ~151,000 TL
- **R² Score**: 0.955

## 🛠️ Geliştirme

### Model Yeniden Eğitme
Modeli yeniden eğitmek için `main.ipynb` dosyasını çalıştırın.

### Yeni Özellik Ekleme
1. Veri setine yeni sütun ekleyin
2. `app.py` dosyasında form alanı oluşturun
3. `prepare_data()` fonksiyonunu güncelleyin

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📞 İletişim

Sorularınız için issue açabilirsiniz.

---

**Not**: Bu tahminler makine öğrenmesi modeli kullanılarak yapılmıştır. Gerçek fiyatlar piyasa koşullarına göre değişebilir.
