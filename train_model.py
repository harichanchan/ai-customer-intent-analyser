import gzip
import json
import joblib
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier

# The five intents our model will recognise
SELECTED_INTENTS = {
    "recover_username",
    "recover_password",
    "reset_2fa",
    "membership",
    "credit_card",
}


# 1. Load the dataset
with gzip.open("abcd_v1.1.json.gz", "rt", encoding="utf-8") as file:
    data = json.load(file)


# 2. Extract the first customer message from each conversation
def prepare_examples(conversations):
    messages = []
    labels = []

    for conversation in conversations:
        for turn in conversation["delexed"]:
            if turn["speaker"] == "customer":
                targets = turn.get("targets", [])
                intent = targets[0] if targets else None
                message = turn.get("text", "").strip()

                if intent in SELECTED_INTENTS and message:
                    messages.append(message)
                    labels.append(intent)

                # Use only the first customer message
                break

    return messages, labels


train_messages, train_labels = prepare_examples(data["train"])
dev_messages, dev_labels = prepare_examples(data["dev"])

print("Training examples:", len(train_messages))
print("Development examples:", len(dev_messages))

print("\nTraining examples per intent:")
print(Counter(train_labels))


# 3. Build the model
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000)
)


# 4. Train using the training set
model.fit(train_messages, train_labels)
# Save the trained model to a file
joblib.dump(model, "intent_model.joblib")
print("\nTrained model saved as intent_model.joblib")

# 5. Evaluate on the development set
predictions = model.predict(dev_messages)

print(
    "\nDevelopment accuracy:",
    round(accuracy_score(dev_labels, predictions), 3)
)

print("\nResults by intent:")
# Compare our model with a simple baseline
baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(train_messages, train_labels)

baseline_predictions = baseline.predict(dev_messages)

print(
    "\nSimple baseline accuracy:",
    round(accuracy_score(dev_labels, baseline_predictions), 3)
)

print(
    "Baseline always predicts:",
    baseline.classes_[baseline.class_prior_.argmax()]
)
print(classification_report(dev_labels, predictions, zero_division=0))


# 6. Show five incorrect predictions
print("\nSOME INCORRECT PREDICTIONS")

shown = 0

for message, actual, predicted in zip(
    dev_messages, dev_labels, predictions
):
    if actual != predicted:
        print("\nCustomer message:", message)
        print("Actual intent:", actual)
        print("Model prediction:", predicted)

        shown += 1

        if shown == 5:
            break


# 7. Show confidence scores for five incorrect predictions
print("\nPREDICTIONS WITH CONFIDENCE")

probabilities = model.predict_proba(dev_messages)

shown = 0

for message, actual, predicted, scores in zip(
    dev_messages, dev_labels, predictions, probabilities
):
    if actual != predicted:
        confidence = max(scores)

        print("\nCustomer message:", message)
        print("Actual intent:", actual)
        print("Model prediction:", predicted)
        print(f"Model confidence: {confidence:.1%}")

        shown += 1

        if shown == 5:
            break


# 8. Compare confidence thresholds on development data
print("\nCONFIDENCE THRESHOLD ANALYSIS")

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    accepted = 0
    correct = 0

    for actual, predicted, scores in zip(
        dev_labels, predictions, probabilities
    ):
        confidence = max(scores)

        if confidence >= threshold:
            accepted += 1

            if actual == predicted:
                correct += 1

    sent_for_review = len(dev_labels) - accepted
    coverage = accepted / len(dev_labels)

    if accepted > 0:
        accepted_accuracy = correct / accepted
    else:
        accepted_accuracy = 0

    print(f"\nThreshold: {threshold:.0%}")
    print(f"Predictions accepted: {accepted}")
    print(f"Sent for review: {sent_for_review}")
    print(f"Coverage: {coverage:.1%}")
    print(f"Accuracy on accepted predictions: {accepted_accuracy:.1%}")


# 9. Evaluate the chosen 50% threshold on the test set
print("\nFINAL TEST SET EVALUATION")

test_messages, test_labels = prepare_examples(data["test"])

test_predictions = model.predict(test_messages)
test_probabilities = model.predict_proba(test_messages)

threshold = 0.5
accepted = 0
correct = 0

for actual, predicted, scores in zip(
    test_labels, test_predictions, test_probabilities
):
    confidence = max(scores)

    if confidence >= threshold:
        accepted += 1

        if actual == predicted:
            correct += 1

review = len(test_labels) - accepted

print("Total test examples:", len(test_labels))
print(
    "Overall accuracy:",
    round(accuracy_score(test_labels, test_predictions), 3)
)
print("Accepted predictions:", accepted)
print("Sent for review:", review)
print(f"Coverage: {accepted / len(test_labels):.1%}")

if accepted > 0:
    print(f"Accuracy on accepted predictions: {correct / accepted:.1%}")
else:
    print("No predictions met the threshold.")


# 10. Check development messages from outside our five categories
print("\nOUTSIDE-CATEGORY CHECK: DEVELOPMENT SET")

outside_messages = []
outside_labels = []

for conversation in data["dev"]:
    for turn in conversation["delexed"]:
        if turn["speaker"] == "customer":
            targets = turn.get("targets", [])
            intent = targets[0] if targets else None
            message = turn.get("text", "").strip()

            if intent and intent not in SELECTED_INTENTS and message:
                outside_messages.append(message)
                outside_labels.append(intent)

            break

if outside_messages:
    outside_scores = model.predict_proba(outside_messages)
    outside_predictions = model.predict(outside_messages)

    flagged = 0
    accepted_examples = []

    for message, actual, predicted, scores in zip(
        outside_messages,
        outside_labels,
        outside_predictions,
        outside_scores
    ):
        confidence = max(scores)

        if confidence < threshold:
            flagged += 1
        else:
            accepted_examples.append(
                (message, actual, predicted, confidence)
            )

    total = len(outside_messages)
    accepted_count = len(accepted_examples)

    print("Outside-category examples:", total)
    print(f"Flagged for review: {flagged} ({flagged / total:.1%})")
    print(
        f"Accepted as a known category: "
        f"{accepted_count} ({accepted_count / total:.1%})"
    )

    print("\nFIRST FIVE ACCEPTED OUTSIDE-CATEGORY EXAMPLES")

    for message, actual, predicted, confidence in accepted_examples[:5]:
        print("\nCustomer message:", message)
        print("Conversation label:", actual)
        print("Model prediction:", predicted)
        print(f"Confidence score: {confidence:.1%}")

else:
    print("No outside-category examples were found.")


# 11. Try your own messages
print("\nTRY THE MODEL")
print("Type 'quit' to stop.")

while True:
    message = input("\nEnter a customer message: ")

    if message.lower() == "quit":
        break

    if not message.strip():
        print("Please enter a message.")
        continue

    prediction = model.predict([message])[0]
    confidence = max(model.predict_proba([message])[0])

    print("Predicted intent:", prediction)
    print(f"Confidence score: {confidence:.1%}")

    if confidence < threshold:
        print("Decision: Ask for more information or send for review.")
    else:
        print("Decision: Accept the predicted intent.")