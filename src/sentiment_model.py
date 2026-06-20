import joblib
import numpy as np
try:
    # Preferred import path
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    try:
        # Fallback (some installs expose the class at package root)
        from vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    except Exception as e:
        raise ImportError(
            "vaderSentiment is required for SentimentModel. Install with 'pip install vaderSentiment'."
        ) from e

_vader = SentimentIntensityAnalyzer()


def _vader_label(text: str) -> dict:
    """Ham metin üzerinde VADER çalıştır."""
    scores = _vader.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        label = 'positive'
    elif compound <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    return {'label': label, 'compound': compound}


def _label_to_proba(label: str) -> dict:
    mapping = {
        'positive': {'negative': 0.05, 'neutral': 0.10, 'positive': 0.85},
        'neutral':  {'negative': 0.15, 'neutral': 0.70, 'positive': 0.15},
        'negative': {'negative': 0.85, 'neutral': 0.10, 'positive': 0.05},
    }
    return mapping.get(label, {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33})


class SentimentModel:
    def __init__(self,
                 model_path='../models/sentiment_model.pkl',
                 vectorizer_path='../models/tfidf_vectorizer.pkl'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, clean_text: str, raw_text: str = None) -> str:
        """
        Hibrit tahmin:
          - clean_text: negation-aware ön işlemden geçmiş metin (ML için)
          - raw_text  : ham metin (VADER için, verilmezse clean_text kullanılır)

        Kısa metinlerde (<8 kelime) VADER ağırlığı artırılır.
        """
        if raw_text is None:
            raw_text = clean_text

        word_count = len(raw_text.split())

        # Ağırlıklar: kısa metin → VADER daha güvenilir
        if word_count < 8:
            ml_w, va_w = 0.35, 0.65
        else:
            ml_w, va_w = 0.55, 0.45

        # ML tahmini
        vec = self.vectorizer.transform([clean_text])
        classes = list(self.model.classes_)
        ml_proba = dict(zip(classes, self.model.predict_proba(vec)[0]))

        # VADER tahmini
        vader_out = _vader_label(raw_text)
        vader_proba = _label_to_proba(vader_out['label'])

        # Birleştir
        all_classes = ['negative', 'neutral', 'positive']
        final = {}
        for cls in all_classes:
            final[cls] = ml_w * ml_proba.get(cls, 0.0) + va_w * vader_proba.get(cls, 0.0)

        prediction = max(final, key=final.get)

        print(f"[ML={self.model.predict(vec)[0]} | VADER={vader_out['label']} | "
              f"Final={prediction} | compound={vader_out['compound']:.3f}]")

        return prediction
