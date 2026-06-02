import os
import pandas as pd
import numpy as np
import re
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------------------------
# 1. Auto-detect project directory (universal fix)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = "depression_dataset_reddit_cleaned.csv"
DATA_PATH = os.path.join(BASE_DIR, DATA_FILE)

if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(
        f"❌ Dataset not found.\nPlace '{DATA_FILE}' inside:\n{BASE_DIR}"
    )

# -------------------------------------------------
# 2. Load Dataset
# -------------------------------------------------
df = pd.read_csv(DATA_PATH, engine="python")
print("✔ Dataset loaded successfully")

# -------------------------------------------------
# 3. NLP Setup
# -------------------------------------------------
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df["clean_text"].apply(clean_text)

# -------------------------------------------------
# 4. Feature Extraction
# -------------------------------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)

X = vectorizer.fit_transform(df["clean_text"])
y = df["is_depression"]

# -------------------------------------------------
# 5. Train/Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------
# 6. Handle Class Imbalance
# -------------------------------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

weights = dict(zip(np.unique(y_train), class_weights))

# -------------------------------------------------
# 7. Train Model
# -------------------------------------------------
model = LinearSVC(class_weight=weights)
model.fit(X_train, y_train)

# -------------------------------------------------
# 8. Evaluation
# -------------------------------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------------------------------
# 9. Save Model & Vectorizer
# -------------------------------------------------
pickle.dump(model, open(os.path.join(BASE_DIR, "model.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(BASE_DIR, "vectorizer.pkl"), "wb"))

print("✔ Model and vectorizer saved successfully")