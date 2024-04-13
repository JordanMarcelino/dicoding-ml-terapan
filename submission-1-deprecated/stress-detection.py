import re

import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from textblob import TextBlob

nltk.download("stopwords")
nltk.download("punkt")
sns.set_style("darkgrid")

df = pd.read_csv(
    "Reddit_Combi.csv",
    sep=";",
    usecols=["Body_Title", "label"],
)

text_lengths = df["Body_Title"].str.len()

# Membuang Outlier
Q1 = text_lengths.quantile(0.25)
Q3 = text_lengths.quantile(0.75)
IQR = Q3 - Q1

df = df[~((text_lengths < (Q1 - 1.5 * IQR)) | (text_lengths > (Q3 + 1.5 * IQR)))]


# Over sampling
num_stres = (df["label"] == 1).sum()

stres = df[df["label"] == 1].sample(1000, random_state=2024)
normal = (
    df[df["label"] == 0].sample(num_stres, replace=True).sample(1000, random_state=2024)
)

over_sampled = pd.concat([stres, normal], axis=0).reset_index(drop=True)

# Data splitting
X = over_sampled["Body_Title"]
y = over_sampled["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=2024, stratify=y
)


# Text preprocessing
class TextPreprocess(TransformerMixin):
    def __init__(self, max_sequence_length: int = 512) -> None:
        super().__init__()
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.max_sequence_length = max_sequence_length

    def remove_noise(self, text: str) -> str:
        # Membuang noise
        patterns = [
            r"[^\w\s]",  # Membuang tanda baca
            r"\d",  # Membuang angka
            r"<.*?>",  # Membuang tag html
            r"http\S+",  # Membuang tautan
            r"\S+@\S+",  # Membuang email
            r"\s+",  # Membuang spasi yang berlebihan
        ]

        for pattern in patterns:
            text = re.sub(pattern, " ", text)

        return text.strip()

    def spell_correction_tokenize(self, text: str) -> list[str]:
        # Memperbaiki ejaan dan tokenisasi teks
        return list(TextBlob(text).correct().tokenize())

    def remove_stopwords(self, text: list[str]) -> list[str]:
        # Membuang stop words
        return [word for word in text if word not in self.stop_words]

    def stemming(self, text: list[str]) -> list[str]:
        # Stemming
        return " ".join([self.stemmer.stem(word) for word in text])

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.Series, y=None):
        return (
            X.str.lower()
            .apply(self.remove_noise)
            .apply(self.spell_correction_tokenize)
            .apply(self.remove_stopwords)
            .apply(self.stemming)
        )


text_pipeline = Pipeline(
    [
        ("text_preprocess", TextPreprocess()),
        ("vectorize", TfidfVectorizer()),
    ]
)

X_train = text_pipeline.fit_transform(X_train)
X_test = text_pipeline.transform(X_test)


# Modeling menggunakan Random Forest
rf = RandomForestClassifier(n_jobs=-1, random_state=2024)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [16, 32],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5],
}

grid = GridSearchCV(
    rf,
    param_grid=param_grid,
    cv=3,
    return_train_score=True,
    n_jobs=-1,
    scoring="accuracy",
    verbose=1,
)

grid.fit(X_train, y_train)


# Evaluasi model
best_estimator = grid.best_estimator_

best_estimator.fit(X_train, y_train)
y_pred = best_estimator.predict(X_test)

final_result = pd.DataFrame(
    {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    },
    index=["RandomForest"],
)

print(final_result)
