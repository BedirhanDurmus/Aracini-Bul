# 🚗 Aracını Bul - Araç Fiyat Tahmin Uygulaması

Bu uygulama, makine öğrenmesi kullanarak araç fiyatlarını tahmin eden kullanıcı dostu bir web arayüzüdür. Gelişmiş filtreleme sistemi ile seçilen marka/model/seriye uygun teknik özellikleri otomatik olarak sunar.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/BedirhanDurmus/Aracini-Bul)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8+-green?logo=python)](https://python.org)

## 🎯 Özellikler

- **🎨 Modern Arayüz**: Streamlit ile responsive ve kullanıcı dostu tasarım
- **🔍 Akıllı Filtreleme**: Seçilen marka/model/seriye göre otomatik filtrelenen seçenekler
- **📊 Yüksek Doğruluk**: XGBoost modeli ile %90+ doğruluk oranı
- **⚡ Gerçek Zamanlı Tahmin**: Anında fiyat tahmini
- **🔧 Kapsamlı Özellikler**: 15+ farklı araç özelliği
- **🎯 Mantıklı Seçimler**: Coupe model seçtiğinizde sadece uygun kasa tipleri görünür
- **📱 Responsive Tasarım**: Mobil ve masaüstü uyumlu
- **🌈 Gelişmiş CSS**: Okunabilir yazılar ve yüksek kontrast

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
git clone https://github.com/BedirhanDurmus/Aracini-Bul.git
cd Aracini-Bul
```

2. **Virtual environment oluşturun (önerilen):**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Gerekli kütüphaneleri yükleyin:**
```bash
pip install -r requirements.txt
```

4. **Uygulamayı çalıştırın:**
```bash
# Yöntem 1: Otomatik çalıştırıcı
python run_app.py

# Yöntem 2: Manuel çalıştırma
streamlit run app.py

# Yöntem 3: Python modülü olarak
python -m streamlit run app.py
```

5. **Tarayıcınızda açın:**
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
Aracini-Bul/
├── app.py                          # Ana Streamlit uygulaması
├── main.ipynb                      # Veri analizi ve model eğitimi
├── cars_tr.csv                     # Araç veri seti (6,675 kayıt)
├── best_car_price_model.pkl        # Eğitilmiş XGBoost modeli
├── model_explainer.py              # Model açıklanabilirlik fonksiyonları
├── run_app.py                      # Otomatik uygulama çalıştırıcı
├── requirements.txt                # Python bağımlılıkları
├── .gitignore                      # Git ignore dosyası
├── pages/
│   └── explainability_page.py      # Model açıklanabilirlik sayfası
├── catboost_info/                  # CatBoost model bilgileri
└── README.md                       # Bu dosya
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

### 🚀 Hızlı Başlangıç
1. **Marka seçin** - Örnek: BMW, Mercedes, Volkswagen
2. **Seri seçin** - Seçilen markaya uygun seriler otomatik filtrelenir
3. **Model seçin** - Seçilen seriye uygun modeller görünür
4. **Teknik özellikler** - Model seçimine göre uygun seçenekler filtrelenir:
   - Vites tipi (sadece o model için mevcut olanlar)
   - Yakıt türü (model için uygun olanlar)
   - Kasa tipi (model ile uyumlu olanlar)
   - Çekiş tipi
5. **Araç durumu** - Kilometre, yıl, motor bilgileri, hasar durumu
6. **Tahmin al** - "Fiyat Tahmini Yap" butonuna tıklayın

### 🎯 Akıllı Filtreleme Sistemi
- **Marka → Seri**: BMW seçtiniz → Sadece BMW serileri görünür
- **Seri → Model**: 3 Serisi seçtiniz → Sadece 3 Serisi modelleri görünür  
- **Model → Teknik**: 320i seçtiniz → Sadece 320i için uygun vites/yakıt/kasa tipleri görünür
- **Mantıklı Seçimler**: Coupe model seçerseniz Hatchback kasa tipi görünmez!

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

## 🔄 Güncellemeler

### v2.0 - Akıllı Filtreleme Sistemi
- ✅ Marka/seri/model bazlı otomatik filtreleme
- ✅ Mantıklı teknik özellik seçimleri
- ✅ Gelişmiş CSS ve okunabilirlik
- ✅ Responsive tasarım iyileştirmeleri

### v1.0 - İlk Sürüm
- ✅ Temel fiyat tahmin sistemi
- ✅ XGBoost modeli entegrasyonu
- ✅ Streamlit arayüzü

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request oluşturun

## 📞 İletişim

- **GitHub**: [BedirhanDurmus](https://github.com/BedirhanDurmus)
- **Repository**: [Aracini-Bul](https://github.com/BedirhanDurmus/Aracini-Bul)
- **Issues**: Sorularınız için issue açabilirsiniz

## 🌟 Demo

Uygulamayı yerel olarak çalıştırarak tüm özellikleri test edebilirsiniz. Canlı demo için repository'yi klonlayın ve yukarıdaki kurulum adımlarını takip edin.

---

**Not**: Bu tahminler makine öğrenmesi modeli kullanılarak yapılmıştır. Gerçek fiyatlar piyasa koşullarına göre değişebilir.
