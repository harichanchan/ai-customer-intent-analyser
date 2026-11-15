# AI Customer Intent Analyser

A machine-learning prototype that classifies the first message
in a customer-support conversation and flags low-confidence
predictions for review.

## Problem

Customer-support teams need to identify what customers are
asking for. Some messages are vague or incomplete, so an
automated classifier may be uncertain or make mistakes.

## Approach

- Used the public Action-Based Conversations Dataset (ABCD).
- Selected five intent categories.
- Trained a TF-IDF and logistic regression model on 1,294
  training conversations.
- Used the development set to select a 50% confidence threshold.
- Built a Streamlit interface for entering customer messages.

## Results

On 144 test examples from the five selected categories:

- Overall accuracy: 72.9%.
- At the 50% threshold, 90 predictions were accepted.
- 89 of those 90 accepted predictions matched the test labels.
- The remaining 54 messages were flagged for review.

## Outside-category analysis

On 847 development examples from other conversation categories,
787 were flagged and 60 were accepted as one of the five known
categories. The labels describe the full conversation, while
the model sees only the first customer message, so some labels
may overlap with the information in that first message.

## Limitations

- The model recognises only five intent categories.
- High confidence does not guarantee a correct prediction.
- Performance was measured on a research dataset, not real
  production traffic.
- The confidence threshold does not reliably detect every
  request outside the known categories.
- This prototype does not send messages to a human reviewer;
  it only displays a review recommendation.

## Dataset

Action-Based Conversations Dataset (ABCD), Chen et al., 2021:
https://aclanthology.org/2021.naacl-main.239/

## How to run the app

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Install the required packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   python -m streamlit run app.py
   ```

The app uses the included `intent_model.joblib` file. It only
loads the model; it does not require the dataset to run.

## How to retrain the model

Download `abcd_v1.1.json.gz` from the
[official ABCD repository](https://github.com/asappresearch/abcd)
and place it in the project folder. Then run:

```bash
python train_model.py
```

The training code also requires the same dependencies listed
in `requirements.txt`.

The model was developed using Python 3.14.3. Reproducing the
saved model may require the same scikit-learn version used to
train it.
