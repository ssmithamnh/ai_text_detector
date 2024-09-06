from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit as st
import torch

# Load model and tokenizer from saved files
model = AutoModelForSequenceClassification.from_pretrained("ai_detector")
tokenizer = AutoTokenizer.from_pretrained("tokenizer")

# Set the model to evaluation mode
model.eval()

# Streamlit App
st.title("AI vs Human Text Classifier")
st.write("This app predicts if the text is AI-generated or human-written.")

# Input text box
input_text = st.text_area("Enter text to classify:")

# Run inference when the user clicks the button
if st.button("Classify"):
    if input_text:
        # Tokenize input text
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

        # Get predictions from the model
        with torch.no_grad():  # Turn off gradient calculation
            outputs = model(**inputs)
        
        # Get logits (raw predictions) and apply softmax to get probabilities
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Get the predicted class (AI-generated or human-written)
        predicted_label = torch.argmax(probabilities, dim=-1).item()

        # Display the prediction result
        if predicted_label == 0:
            st.write("This text is **Human-written**.")
        else:
            st.write("This text is **AI-generated**.")
    else:
        st.write("Please enter some text.")


