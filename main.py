# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import re

# Constants
MAX_LEN = 500
MAX_VOCAB_SIZE = 10000

# Load the IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.keras')

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    # Clean and tokenize
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and convert to lowercase
    words = text.split()
    
    # Encode the words while ensuring out-of-vocabulary words get assigned index `1`
    encoded_review = [word_index.get(word, 1) for word in words]  # 1 for out-of-vocabulary words
    
    # Ensure the review is padded to the max length
    padded_review = sequence.pad_sequences([encoded_review], maxlen=MAX_LEN)
    return padded_review

# Step 3: Streamlit App
st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review and classify it as **Positive** or **Negative** based on sentiment.')

# User input
user_input = st.text_area('✍️ Movie Review:')

if st.button('Classify'):
    if user_input.strip() == '':
        st.warning('Please enter a movie review.')
    else:
        preprocessed_input = preprocess_text(user_input)
        
        # Ensure the input has the correct shape and data type
        preprocessed_input = np.array(preprocessed_input, dtype=np.float32)

        # Make prediction
        try:
            prediction = model.predict(preprocessed_input)
            sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
            score = float(prediction[0][0])

            st.subheader('🔍 Result')
            st.write(f'**Sentiment:** {sentiment}')
            st.write(f'**Prediction Score:** {score:.4f}')
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info('Waiting for input...')
