import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Set page configuration
st.set_page_config(
    page_title="Patient Emotion Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define emotion emojis mapping
emotion_emojis = {
    'anxiety': '😰',
    'stress': '😓',
    'confusion': '😕',
    'hopeful': '🤞',
    'fear': '😨'
}

# Define emotion descriptions
emotion_descriptions = {
    'anxiety': 'Feelings of worry, nervousness, or unease about a situation with an uncertain outcome.',
    'stress': 'State of mental or emotional strain resulting from adverse or demanding circumstances.',
    'confusion': 'Lack of understanding or uncertainty about a situation, concept, or instructions.',
    'hopeful': 'Feeling of expectation and desire for something to happen or become true.',
    'fear': 'An unpleasant emotion caused by the threat of danger, pain, or harm.'
}

# Function to load model
@st.cache_resource
def load_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model directory
    model_dir = os.path.join(os.path.dirname(__file__), 'models/transformers_patient_sentiment')
    
    # Load tokenizer and model from local directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # Move model to device
    model = model.to(device)
    
    # Define emotional states
    emotional_states = ['anxiety', 'stress', 'confusion', 'hopeful', 'fear']
    
    return model, tokenizer, device, emotional_states

# Function to predict emotions
def predict_emotions(text, model, tokenizer, device, emotional_states):
    model.eval()
    
    # Tokenize text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=32,  # Using smaller max length for faster processing
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    # Move inputs to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # Get probabilities
        predictions = (probs >= 0.5).astype(int)  # Apply threshold
    
    # Create results dictionary
    results = {}
    for i, emotion in enumerate(emotional_states):
        results[emotion] = {
            'present': bool(predictions[i]),
            'probability': float(probs[i]),
            'emoji': emotion_emojis.get(emotion, ''),
            'description': emotion_descriptions.get(emotion, '')
        }
    
    return results

# Main app function
def main():
    # Title and description
    st.title("Patient Emotion Analyzer 🧠")
    st.markdown("""
    This app uses a transformer-based model to analyze patient sentiments and extract emotional states.
    Enter a patient's sentiment text below to see which emotions are detected.    """)
    
    # Load model, tokenizer, and states
    with st.spinner("Loading model... This may take a moment."):
        model, tokenizer, device, emotional_states = load_model()
    
    # Input area
    st.subheader("Enter Patient Sentiment")
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    
    input_text = st.text_area(
        "Type or paste patient's sentiment here:",
        value=st.session_state.text_input,
        height=150,
        placeholder="Example: Patient expresses anxiety about upcoming procedure and confusion regarding medication instructions."
    )
    
    # Sample prompts
    st.markdown("**Or try one of these examples:**")
    examples = [
        "Patient is hopeful and shows no significant anxiety, stress, or fear related to health conditions.",
        "Patient expresses fear and anxiety about high blood pressure and possible complications.",
        "Elderly patient expresses fear of declining health, confusion about medications, and stress related to mobility issues.",
        "Patient (minor) is anxious and fearful about medical procedures, sometimes confused by instructions, and stressed by separation from family."
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Example 1"):
            st.session_state.text_input = examples[0]
            st.rerun()
        if st.button("Example 3"):
            st.session_state.text_input = examples[2]
            st.rerun()
    
    with col2:
        if st.button("Example 2"):
            st.session_state.text_input = examples[1]
            st.rerun()
        if st.button("Example 4"):
            st.session_state.text_input = examples[3]
            st.rerun()
    
    # If session state has input text, use it
    if 'input_text' in st.session_state:
        input_text = st.session_state.input_text
        # Clear the session state
        del st.session_state.input_text
        
    # Add custom CSS for red button
    st.markdown("""
    <style>
    div.stButton > button[kind="primary"] {
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Predict button
    if st.button("Analyze Emotions", type="primary", use_container_width=True, key="analyze_button"):
        if input_text:
            with st.spinner("Analyzing emotions..."):
                # Get predictions
                results = predict_emotions(input_text, model, tokenizer, device, emotional_states)
                
                # Display results
                st.subheader("Detected Emotions")
                
                # Create two columns - one for detected, one for not detected
                detected_col, not_detected_col = st.columns(2)
                
                detected_emotions = []
                not_detected_emotions = []
                
                for emotion, result in results.items():
                    if result['present']:
                        detected_emotions.append((emotion, result))
                    else:
                        not_detected_emotions.append((emotion, result))
                
                with detected_col:
                    st.markdown("### Emotions Present ✅")
                    if detected_emotions:
                        for emotion, result in detected_emotions:
                            with st.expander(f"{result['emoji']} {emotion.capitalize()} - {result['probability']:.2f}"):
                                st.markdown(f"**Description**: {result['description']}")
                                st.progress(result['probability'])
                    else:
                        st.write("No emotions detected.")
                
                with not_detected_col:
                    st.markdown("### Emotions Not Present ❌")
                    if not_detected_emotions:
                        for emotion, result in not_detected_emotions:
                            with st.expander(f"{result['emoji']} {emotion.capitalize()} - {result['probability']:.2f}"):
                                st.markdown(f"**Description**: {result['description']}")
                                st.progress(result['probability'])
                    else:
                        st.write("All emotions detected!")
                
                # Visual representation of emotions
                st.subheader("Emotion Probability Chart")
                
                # Create a DataFrame for the chart
                chart_data = pd.DataFrame({
                    'Emotion': [f"{emotion_emojis[emotion]} {emotion.capitalize()}" for emotion in emotional_states],
                    'Probability': [results[emotion]['probability'] for emotion in emotional_states]
                })
                
                # Sort by probability
                chart_data = chart_data.sort_values('Probability', ascending=False)
                
                # Display the chart
                st.bar_chart(chart_data.set_index('Emotion'))
                
                # Show a summary
                st.subheader("Summary")
                emotion_summary = [f"{emotion_emojis[emotion]} {emotion.capitalize()}" 
                                 for emotion in emotional_states 
                                 if results[emotion]['present']]
                
                if emotion_summary:
                    st.markdown(f"The patient is expressing: **{', '.join(emotion_summary)}**")
                else:
                    st.markdown("No significant emotions detected in the text.")
        else:
            st.warning("Please enter some text to analyze.")
    
    # Add information about the model
    with st.sidebar:
        st.header("About the Model")
        st.markdown("""
        This app uses a TinyBERT model fine-tuned for multi-label emotion classification in patient sentiments.
        
        **Emotional states detected:**
        - 😰 Anxiety
        - 😓 Stress
        - 😕 Confusion
        - 🤞 Hopeful
        - 😨 Fear
        
        **Model architecture:**
        - Base model: TinyBERT (4.4M parameters)
        - Classification type: Multi-label
        - Input sequence length: 32 tokens
        
        This lightweight model is optimized for CPU inference and can be deployed in resource-constrained environments.
        """)
        
        st.markdown("---")
        st.markdown("Created for AAI-510 - Machine Learning Fundamentals and Applications")

# Run the app
if __name__ == "__main__":
    main()
