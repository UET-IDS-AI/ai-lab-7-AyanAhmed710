import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


# =========================
# Q1 Naive Bayes
# =========================

def naive_bayes_mle_spam():
    texts = [
        "win money now",
        "limited offer win cash",
        "cheap meds available",
        "win big prize now",
        "exclusive offer buy now",
        "cheap pills buy cheap meds",
        "win lottery claim prize",
        "urgent offer win money",
        "free cash bonus now",
        "buy meds online cheap",
        "meeting schedule tomorrow",
        "project discussion meeting",
        "please review the report",
        "team meeting agenda today",
        "project deadline discussion",
        "review the project document",
        "schedule a meeting tomorrow",
        "please send the report",
        "discussion on project update",
        "team sync meeting notes"
    ]

    labels = np.array([
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0,0
    ])

    test_email = "win cash prize now"

    # Tokenize
    tokenized = [text.split() for text in texts]

    # Vocabulary
    vocab = set()
    for tokens in tokenized:
        vocab.update(tokens)
    vocab = list(vocab)

    # Priors
    priors = {}
    for c in [0, 1]:
        priors[c] = np.mean(labels == c)

    # Word counts per class
    word_counts = {0: {}, 1: {}}
    total_words = {0: 0, 1: 0}

    for c in [0, 1]:
        word_counts[c] = {word: 0 for word in vocab}

    for tokens, label in zip(tokenized, labels):
        for word in tokens:
            word_counts[label][word] += 1
            total_words[label] += 1

    # Word probabilities (MLE, no smoothing)
    word_probs = {0: {}, 1: {}}
    for c in [0, 1]:
        for word in vocab:
            if total_words[c] > 0:
                word_probs[c][word] = word_counts[c][word] / total_words[c]
            else:
                word_probs[c][word] = 0.0

    # Prediction
    test_tokens = test_email.split()

    scores = {}
    for c in [0, 1]:
        score = np.log(priors[c])
        for word in test_tokens:
            prob = word_probs[c].get(word, 0)

            # Avoid log(0) → treat unseen as very small probability
            if prob == 0:
                score += -1e9
            else:
                score += np.log(prob)

        scores[c] = score

    prediction = max(scores, key=scores.get)

    return priors, word_probs, prediction


# =========================
# Q2 KNN
# =========================

def knn_iris(k=3, test_size=0.2, seed=0):
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Euclidean distance
    def euclidean_distance(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # Predict function
    def predict(X1, X2, y2):
        predictions = []
        for x in X1:
            distances = [euclidean_distance(x, x_train) for x_train in X2]
            k_indices = np.argsort(distances)[:k]
            k_labels = y2[k_indices]

            # Majority voting
            counts = np.bincount(k_labels)
            pred = np.argmax(counts)
            predictions.append(pred)

        return np.array(predictions)

    # Predictions
    train_predictions = predict(X_train, X_train, y_train)
    test_predictions = predict(X_test, X_train, y_train)

    # Accuracy
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    return train_accuracy, test_accuracy, test_predictions