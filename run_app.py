#!/usr/bin/env python3
"""
Araç Fiyat Tahmin Uygulaması Çalıştırıcı
"""

import subprocess
import sys
import os

def check_requirements():
    """Gerekli paketleri kontrol et"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn', 
        'xgboost', 'joblib', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Eksik paketler: {', '.join(missing_packages)}")
        print("📦 Paketleri yüklemek için: pip install -r requirements.txt")
        return False
    
    print("✅ Tüm gerekli paketler mevcut")
    return True

def check_files():
    """Gerekli dosyaları kontrol et"""
    required_files = [
        'app.py',
        'cars_tr.csv',
        'best_car_price_model.pkl'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Eksik dosyalar: {', '.join(missing_files)}")
        return False
    
    print("✅ Tüm gerekli dosyalar mevcut")
    return True

def run_app():
    """Uygulamayı çalıştır"""
    try:
        print("🚀 Uygulama başlatılıyor...")
        print("🌐 Tarayıcınızda http://localhost:8501 adresini açın")
        print("⏹️  Durdurmak için Ctrl+C tuşlayın")
        print("-" * 50)
        
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', '8501',
            '--server.address', 'localhost'
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Uygulama durduruldu")
    except Exception as e:
        print(f"❌ Hata: {e}")

def main():
    """Ana fonksiyon"""
    print("🚗 Araç Fiyat Tahmin Uygulaması")
    print("=" * 40)
    
    # Kontroller
    if not check_requirements():
        return
    
    if not check_files():
        return
    
    # Uygulamayı çalıştır
    run_app()

if __name__ == "__main__":
    main()
