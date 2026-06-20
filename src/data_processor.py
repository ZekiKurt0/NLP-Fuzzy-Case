import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def init_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)


NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none",
    "nobody", "nothing", "nowhere", "hardly", "barely",
    "scarcely", "n't"
}

NEGATION_WINDOW = 4   


class DataProcessor:
    def __init__(self):
        init_nltk()
     
        base_stops = set(stopwords.words('english'))
        self.stop_words = base_stops - NEGATION_WORDS
        self.lemmatizer = WordNetLemmatizer()

    def _apply_negation(self, text: str) -> str:
        """
        Negation belirtecinden sonraki NEGATION_WINDOW kelimeyi
        NOT_ önekiyle işaretler.
        Örnek: "not good at all" → "not NOT_good NOT_at NOT_all"
        """
        tokens = text.lower().split()
        result = []
        negate = 0

        for tok in tokens:
            
            if re.search(r'[.!?,;]', tok):
                negate = 0
                result.append(tok)
                continue

            if tok in NEGATION_WORDS or tok.endswith("n't"):
                negate = NEGATION_WINDOW
                result.append(tok)
            elif negate > 0:
                result.append(f"NOT_{tok}")
                negate -= 1
            else:
                result.append(tok)

        return ' '.join(result)

    def clean_text(self, text: str) -> str:
        """
        Adımlar:
          1. Negation işaretleme  (not good → NOT_good)
          2. Küçük harf + URL/noktalama temizleme (NOT_ alt çizgisi korunur)
          3. Stop-word kaldırma   (negation kelimeleri hariç)
          4. Lemmatization        (NOT_ öneki korunur)
        """
        if not isinstance(text, str):
            return ""

        
        text = self._apply_negation(text)

       
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-z_\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        
        tokens = text.split()
        processed = []
        for w in tokens:
            if len(w) <= 1:
                continue
            if w.startswith('not_'):
                
                root = self.lemmatizer.lemmatize(w[4:])
                processed.append(f'not_{root}')
            elif w not in self.stop_words:
                processed.append(self.lemmatizer.lemmatize(w))

        return " ".join(processed)
