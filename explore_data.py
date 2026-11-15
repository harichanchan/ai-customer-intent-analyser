import gzip
import json

# Read the compressed dataset
with gzip.open("abcd_v1.1.json.gz", "rt", encoding="utf-8") as file:
    data = json.load(file)

# Count conversations in each section
print("DATASET SECTIONS")
for section in data:
    print(f"{section}: {len(data[section])} conversations")

# Look at the first conversation
conversation = data["train"][0]

print("\nFIRST CONVERSATION")
print("Information available:", list(conversation.keys()))

# Print only the first few customer messages and their labels
print("\nFIRST CUSTOMER MESSAGES")

shown = 0

for turn in conversation["delexed"]:
    if turn["speaker"] == "customer":
        print("Message:", turn["text"])
        print("Intent:", turn["targets"][0])
        print()

        shown += 1
        if shown == 3:
            break
from collections import Counter

# Count each conversation's intent
intent_counts = Counter()

for conversation in data["train"]:
    for turn in conversation["delexed"]:
        if turn["speaker"] == "customer" and turn["targets"][0]:
            intent = turn["targets"][0]
            intent_counts[intent] += 1
            break

print("\nMOST COMMON INTENTS IN TRAINING DATA")

for intent, count in intent_counts.most_common(10):
    print(f"{intent}: {count}")
print("\nEXAMPLES FROM COMMON INTENTS")

# Store the ten labels we want to inspect
common_intents = [intent for intent, count in intent_counts.most_common(10)]
shown_intents = set()

for conversation in data["train"]:
    customer_turns = [
        turn
        for turn in conversation["delexed"]
        if turn["speaker"] == "customer"
        and turn["targets"]
        and turn["targets"][0]
    ]

    if not customer_turns:
        continue

    intent = customer_turns[0]["targets"][0]

    # Show only one conversation for each selected intent
    if intent in common_intents and intent not in shown_intents:
        print(f"\nIntent: {intent}")

        for turn in customer_turns[:2]:
            print("Customer:", turn["text"])

        shown_intents.add(intent)

        if len(shown_intents) == len(common_intents):
            break            