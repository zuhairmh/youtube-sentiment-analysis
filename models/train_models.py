import pandas as pd
import joblib

from scipy.sparse import hstack

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from tqdm import tqdm



# ==================================================
# LOAD DATASET
# ==================================================

print("\nLoading dataset...")

df = pd.read_csv("../datasets/clean_data.csv")

print(f"Dataset Size: {len(df)} comments")


# ==================================================
# FEATURES + LABELS
# ==================================================

X = df["Cleaned_Comment"]
y = df["Sentiment"]


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==================================================
# WORD TF-IDF
# ==================================================

print("\nBuilding WORD TF-IDF features...")

word_vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.90,
    max_features=150000,
    sublinear_tf=True
)

X_train_word = word_vectorizer.fit_transform(X_train)

X_test_word = word_vectorizer.transform(X_test)

print(f"Word Feature Count: {X_train_word.shape[1]}")


# ==================================================
# CHARACTER TF-IDF
# ==================================================

print("\nBuilding CHARACTER TF-IDF features...")

char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(3, 4),
    min_df=50,
    max_df=0.90,
    max_features=100000,
    sublinear_tf=True
)

X_train_char = char_vectorizer.fit_transform(X_train)

X_test_char = char_vectorizer.transform(X_test)

print(f"Character Feature Count: {X_train_char.shape[1]}")


# ==================================================
# COMBINE FEATURES
# ==================================================

print("\nCombining word + character features...")

X_train_combined = hstack([
    X_train_word,
    X_train_char
])

X_test_combined = hstack([
    X_test_word,
    X_test_char
])

print(f"Final Feature Count: {X_train_combined.shape[1]}")


# ==================================================
# MODELS
# ==================================================

models = {

    "Naive_Bayes": MultinomialNB(),

    "Logistic_Regression": LogisticRegression(
        max_iter=1000,
        solver='saga',
        class_weight='balanced',
        n_jobs=-1,
        verbose=1
    ),

    "Linear_SVM": LinearSVC(
        class_weight='balanced',
        verbose=1
    ),

    "SGD_Classifier": SGDClassifier(
        loss='hinge',
        max_iter=2000,
        class_weight='balanced',
        verbose=1
    )
}


# ==================================================
# TRAIN + EVALUATE
# ==================================================

results = {}

for name, model in tqdm(
    models.items(),
    desc="Training Models",
    unit="model"
):

    print("\n=================================================")
    print(f"TRAINING: {name}")
    print("=================================================")

    # TRAIN MODEL
    model.fit(
        X_train_combined,
        y_train
    )

    # PREDICT
    predictions = model.predict(
        X_test_combined
    )

    # ACCURACY
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = accuracy

    print(f"\nAccuracy: {accuracy:.4f}")

    # CLASSIFICATION REPORT
    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    # SAVE MODEL
    model_filename = f"{name}_model.pkl"

    joblib.dump(
        model,
        model_filename
    )

    print(f"\nSaved Model: {model_filename}")


# ==================================================
# SAVE VECTORIZERS
# ==================================================

print("\nSaving vectorizers...")

joblib.dump(
    word_vectorizer,
    "word_tfidf_vectorizer.pkl"
)

joblib.dump(
    char_vectorizer,
    "char_tfidf_vectorizer.pkl"
)


# ==================================================
# FINAL RESULTS
# ==================================================

print("\n=================================================")
print("TRAINING COMPLETE")
print("=================================================")

print("\nMODEL ACCURACIES:\n")

for model_name, accuracy in results.items():

    print(f"{model_name}: {accuracy:.4f}")

print("\nSaved Files:")
print("- Naive_Bayes_model.pkl")
print("- Logistic_Regression_model.pkl")
print("- Linear_SVM_model.pkl")
print("- SGD_Classifier_model.pkl")
print("- word_tfidf_vectorizer.pkl")
print("- char_tfidf_vectorizer.pkl")

