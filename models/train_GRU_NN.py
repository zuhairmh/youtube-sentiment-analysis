import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Embedding,
    GRU,
    Dense,
    Dropout
)

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import (
    EarlyStopping,
    Callback
)



# ==================================================
# CUSTOM TIMER CALLBACK
# ==================================================

class TrainingProgressCallback(Callback):

    def on_train_begin(self, logs=None):

        self.training_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):

        self.epoch_start = time.time()

        print("\n=================================================")
        print(f"STARTING EPOCH {epoch + 1}")
        print("=================================================")

    def on_epoch_end(self, epoch, logs=None):

        epoch_time = time.time() - self.epoch_start

        total_elapsed = time.time() - self.training_start

        remaining_epochs = self.params['epochs'] - (epoch + 1)

        estimated_remaining = epoch_time * remaining_epochs

        print("\n-------------------------------------------------")
        print(f"Epoch Completed: {epoch + 1}/{self.params['epochs']}")

        print(f"Training Accuracy: {logs['accuracy']:.4f}")

        print(f"Validation Accuracy: {logs['val_accuracy']:.4f}")

        print(f"Epoch Time: {epoch_time / 60:.2f} minutes")

        print(f"Total Elapsed: {total_elapsed / 60:.2f} minutes")

        print(
            f"Estimated Time Remaining: "
            f"{estimated_remaining / 60:.2f} minutes"
        )

        print("-------------------------------------------------")


# ==================================================
# LOAD DATASET
# ==================================================

print("\nLoading dataset...")

df = pd.read_csv("../datasets/clean_data.csv")

print(f"Dataset Size: {len(df)} comments")


# ==================================================
# FEATURES + LABELS
# ==================================================

X = df["Cleaned_Comment"].astype(str)

y = df["Sentiment"]


# ==================================================
# ENCODE LABELS
# ==================================================

print("\nEncoding labels...")

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

y_categorical = to_categorical(y_encoded)


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42
)


# ==================================================
# TOKENIZATION
# ==================================================

print("\nTokenizing text...")

MAX_WORDS = 20000

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)

X_test_seq = tokenizer.texts_to_sequences(X_test)

print("Tokenization complete.")


# ==================================================
# PADDING
# ==================================================

MAX_LENGTH = 50

print("\nPadding sequences...")

X_train_pad = pad_sequences(
    X_train_seq,
    maxlen=MAX_LENGTH,
    padding='post',
    truncating='post'
)

X_test_pad = pad_sequences(
    X_test_seq,
    maxlen=MAX_LENGTH,
    padding='post',
    truncating='post'
)

print("Padding complete.")


# ==================================================
# BUILD MODEL
# ==================================================

print("\nBuilding GRU model...")

model = Sequential([

    Embedding(
        input_dim=MAX_WORDS,
        output_dim=64,
        input_length=MAX_LENGTH
    ),

    GRU(
        32,
        dropout=0.2,
        recurrent_dropout=0.2
    ),

    Dense(
        64,
        activation='relu'
    ),

    Dropout(0.3),

    Dense(
        3,
        activation='softmax'
    )
])


# ==================================================
# COMPILE MODEL
# ==================================================

print("\nCompiling model...")

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# ==================================================
# MODEL SUMMARY
# ==================================================

print("\nMODEL SUMMARY:\n")

model.summary()


# ==================================================
# CALLBACKS
# ==================================================

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

progress_callback = TrainingProgressCallback()


# ==================================================
# TRAIN MODEL
# ==================================================

print("\n=================================================")
print("STARTING TRAINING")
print("=================================================")

history = model.fit(

    X_train_pad,
    y_train,

    validation_split=0.1,

    epochs=5,

    batch_size=16,

    callbacks=[
        early_stopping,
        progress_callback
    ],

    verbose=1
)


# ==================================================
# EVALUATE MODEL
# ==================================================

print("\nEvaluating model...")

predictions = model.predict(X_test_pad)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

true_classes = np.argmax(
    y_test,
    axis=1
)

accuracy = accuracy_score(
    true_classes,
    predicted_classes
)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=label_encoder.classes_
    )
)


# ==================================================
# SAVE MODEL
# ==================================================

print("\nSaving model...")

model.save("gru_sentiment_model.keras")


# ==================================================
# SAVE TOKENIZER + LABEL ENCODER
# ==================================================

joblib.dump(
    tokenizer,
    "gru_tokenizer.pkl"
)

joblib.dump(
    label_encoder,
    "label_encoder.pkl"
)


# ==================================================
# DONE
# ==================================================

print("\n=================================================")
print("TRAINING COMPLETE")
print("=================================================")

print("\nSaved Files:")
print("- gru_sentiment_model.keras")
print("- gru_tokenizer.pkl")
print("- label_encoder.pkl")
