import joblib
import streamlit as st


# Load the trained model
model = joblib.load("intent_model.joblib")

# Minimum score required to accept a prediction
THRESHOLD = 0.5

# Display names for our five categories
intent_names = {
    "recover_username": "Username recovery",
    "recover_password": "Password recovery",
    "reset_2fa": "Reset two-factor authentication",
    "membership": "Membership",
    "credit_card": "Credit card issue",
}

st.title("AI Customer Intent Analyser")

st.write(
    "Enter a customer message to see the predicted intent "
    "and whether it should be reviewed."
)

message = st.text_area(
    "Customer message",
    placeholder="For example: I forgot my password"
)

if st.button("Analyse message"):
    if not message.strip():
        st.warning("Please enter a customer message.")
    else:
        prediction = model.predict([message])[0]
        confidence = max(model.predict_proba([message])[0])

        st.subheader("Result")
        st.write("**Predicted intent:**", intent_names[prediction])
        st.write(f"**Model confidence score:** {confidence:.1%}")

        if confidence < THRESHOLD:
            st.warning("Low confidence: ask for more information or review.")
        else:
            st.success("The prediction meets the 50% confidence threshold.")

        st.caption(
            "Research demo: the model only recognises five intents. "
            "A high score does not guarantee a correct prediction, "
            "especially for requests outside those categories."
        )