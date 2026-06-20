Markdown
# 🚀 Müşteri Yorumu Analiz Sistemi (NLP & Fuzzy Logic)

Bu proje, e-ticaret platformlarındaki müşteri yorumlarını otomatik olarak analiz etmek, duygu sınıflaması yapmak ve bulanık mantık (Fuzzy Logic) tabanlı güvenilirlik skorları üretmek amacıyla geliştirilmiş entegre bir yapay zeka çözümüdür.

## 📋 Proje Özeti
* **Duygu Analizi:** Makine öğrenmesi algoritmaları ile yorumların olumlu, olumsuz veya nötr olup olmadığını belirler.
* **Güvenilirlik Skorlama:** Yorumun uzunluğu, ürün puanı ve yayınlanma tarihi gibi parametreleri kullanarak "yorum güvenilirliği" hesaplar.
* **İş Odaklı Pipeline:** Analiz edilen veriler üzerinden otomatik "şikayet özeti" üretimine olanak tanıyan bir altyapı sunar.

## 📂 Teknik Mimari ve Dosya Yapısı
Sistem, sürdürülebilir ve modüler bir yapı üzerine inşa edilmiştir:

```text
piton-nlp-fuzzy-case/
├── app/
│   └── main.py             # Streamlit tabanlı interaktif kullanıcı arayüzü
├── data/
│   ├── raw/                # Orijinal Kaggle veri seti (7817_1.csv)
│   └── processed/          # Temizlenmiş ve dengelenmiş veriler
├── models/                 # Eğitilmiş .pkl modelleri ve vektörelleştiriciler
├── notebooks/              # Adım adım veri analizi ve modelleme süreçleri
└── src/                    # Modüler kaynak kodları
    ├── data_processor.py   # NLP ön işleme (Regex, Lemmatization, Stop-words)
    ├── sentiment_model.py  # Tahmin motoru (Random Forest/Logistic Regression)
    └── fuzzy_inference.py  # Scikit-fuzzy tabanlı güvenilirlik sistemi
🛠 Teknik Tercihler ve Gerekçeler
NLP Ön İşleme: Metin temizliğinde NLTK kütüphanesi kullanılarak; lowercasing, punctuation removal, stop-word elimination ve lemmatization adımları uygulandı.

Vektörizasyon: Yüksek boyutlu metin verilerinde seyrek matris (sparse matrix) oluşumunu ve anlamsal ayırt ediciliği optimize etmek için TF-IDF yöntemi seçildi.

Model: Random Forest Classifier, metin verisindeki doğrusal olmayan (non-linear) karmaşık örüntüleri yakalamadaki başarısı ve hiperparametre optimizasyonu ile sağlanan genel başarımı nedeniyle ana model olarak seçildi.

Bulanık Mantık: Yorum güvenilirliği gibi sübjektif ve belirsizlik içeren kavramları matematiksel olarak modellemek için scikit-fuzzy kullanıldı.

🚀 Kurulum (Setup)
Repoyu Klonlayın:

Bash
git clone [REPONUN_LINKI]
cd piton-nlp-fuzzy-case
Sanal Ortam Oluşturun:

Bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
Bağımlılıkları Yükleyin:

Bash
pip install -r requirements.txt
Uygulamayı Çalıştırın:

Bash
streamlit run app/main.py



Geliştirici: [Muhammed Zeki Kurt] İletişim: [m.zekikurtt@gmail.com]