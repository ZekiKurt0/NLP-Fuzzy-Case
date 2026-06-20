import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processor import DataProcessor
from src.sentiment_model import SentimentModel
from src.fuzzy_inference import get_reliability_score

import streamlit as st

st.set_page_config(page_title="Duygu Analiz Sistemi", page_icon="🤖")

st.title("🤖 Müşteri Yorumu Analiz Sistemi")
st.markdown("E-ticaret yorumlarını analiz edin ve güvenilirlik skorunu görün.")


@st.cache_resource
def load_components():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'sentiment_model.pkl')
    vec_path   = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl')
    return DataProcessor(), SentimentModel(model_path=model_path, vectorizer_path=vec_path)


processor, model = load_components()

user_input = st.text_area("Yorumunuzu buraya yazın(İngilizce):", height=150)
rating     = st.slider("Ürün Puanı (1-5):", 1, 5, 3)
review_age = st.number_input("Yorumun üzerinden geçen gün sayısı:", min_value=0, value=30)

if st.button("Analiz Et"):
    if user_input.strip() == "":
        st.warning("Lütfen bir yorum girin!")
    else:
        # 1. NLP İşleme (negation-aware)
        cleaned_text = processor.clean_text(user_input)

        # 2. Duygu Tahmini (hibrit: ML + VADER)
        #    raw_text → VADER için ham metin
        #    cleaned_text → ML modeli için ön işlenmiş metin
        sentiment = model.predict(clean_text=cleaned_text, raw_text=user_input)

        # 3. Güvenilirlik Skoru
        length = len(user_input.split())
        reliability = get_reliability_score(rating, length, review_age)

        # Sonuçları Göster
        col1, col2 = st.columns(2)

        sentiment_emoji = {"positive": "😊 Olumlu", "neutral": "😐 Nötr", "negative": "😞 Olumsuz"}
        col1.metric("Duygu Durumu", sentiment_emoji.get(sentiment, sentiment))
        col2.metric("Güvenilirlik Skoru", f"{reliability:.1f} / 100")

        with st.expander("🔬 Detaylar"):
            st.write("**Ön İşlenmiş Metin:**", cleaned_text)

        st.success("Analiz tamamlandı!")

