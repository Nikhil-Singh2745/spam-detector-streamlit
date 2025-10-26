import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re # Still useful for analysis here
import string # To help with word counting/cleaning for keyword check

# --- Load Saved Models ---
# These models were trained WITHOUT NLTK preprocessing
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'vectorizer.pkl' not found. Make sure it's in the same directory as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading vectorizer.pkl: {e}")
    st.stop()

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Make sure it's in the same directory as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()

# --- Common Spam Keywords (Case-Insensitive) ---
SPAM_KEYWORDS = [
    "free", "win", "winner", "prize", "claim", "urgent", "cash", "offer",
    "limited", "won", "$", "£", "click", "subscribe", "buy", "cheap",
    "sex", "xxx", "dating", "bonus", "guaranteed", "credit", "loan",
    "income", "earn", "extra", "money", "apply", "online", "now", "call"
]

# --- Helper Function for Text Analysis ---
def analyze_text(text):
    """Calculates basic text features."""
    if not text or not isinstance(text, str): # Added type check
        return 0, 0, 0.0, 0.0, []

    length = len(text)
    words = text.split()
    word_count = len(words)
    upper_count = sum(1 for char in text if char.isupper())
    digit_count = sum(1 for char in text if char.isdigit())

    percent_upper = (upper_count / length * 100) if length > 0 else 0
    percent_digits = (digit_count / length * 100) if length > 0 else 0

    # Clean text for keyword matching (lowercase, remove punctuation)
    cleaned_text = text.lower().translate(str.maketrans('', '', string.punctuation))
    # Check word by word for keyword presence
    found_keywords = [keyword for keyword in SPAM_KEYWORDS if keyword in cleaned_text.split()]

    return length, word_count, percent_upper, percent_digits, list(set(found_keywords)) # Use set for unique keywords


# --- Streamlit App Interface ---
st.set_page_config(
    page_title="Spam Detector AI",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="auto"
)

# Custom CSS (same as before)
st.markdown("""
    <style>
        /* General body styling */
        .main {
            background-color: #f0f8ff; /* Light Alice Blue background */
            padding: 2rem;
        }
        /* Title styling */
        h1 {
            color: #4682b4; /* Steel Blue */
            text-align: center;
            font-family: 'Arial Black', Gadget, sans-serif;
        }
        /* Text area styling */
        .stTextArea textarea {
            background-color: #ffffff;
            border: 2px solid #add8e6; /* Light Blue border */
            border-radius: 10px;
            color: #333;
            font-size: 16px;
        }
        /* Button styling */
        .stButton button {
            background-color: #5f9ea0; /* Cadet Blue */
            color: white;
            padding: 12px 28px;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            display: block;
            margin: 20px auto; /* Center button */
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #4682b4; /* Steel Blue on hover */
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        /* Example buttons styling */
        div[data-testid="stHorizontalBlock"] > div[data-testid^="stVerticalBlock"] .stButton button {
             margin: 5px auto; /* Less margin for example buttons */
             padding: 8px 18px;
             font-size: 14px;
             background-color: #b0c4de; /* Light Steel Blue */
        }
         div[data-testid="stHorizontalBlock"] > div[data-testid^="stVerticalBlock"] .stButton button:hover {
              background-color: #a9a9a9; /* Dark Gray */
         }
        /* Result styling */
        .stAlert {
            border-radius: 8px;
            font-size: 16px;
            text-align: center;
        }
        .stSuccess {
             border: 2px solid #2e8b57; /* Sea Green border */
        }
        .stError {
             border: 2px solid #cd5c5c; /* Indian Red border */
        }
        /* Progress bar styling */
        .stProgress > div > div > div > div {
            background-color: #5f9ea0; /* Cadet Blue */
        }
        /* Footer styling */
        footer {
            visibility: hidden; /* Hide default Streamlit footer */
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #e6e6fa; /* Lavender */
            color: #4682b4; /* Steel Blue */
            text-align: center;
            padding: 10px;
            font-size: 14px;
        }
        /* Style for keywords */
        .keyword {
             background-color: #ffe4e1; /* Misty Rose */
             padding: 2px 5px;
             border-radius: 3px;
             margin: 2px;
             display: inline-block;
             font-family: monospace;
             color: #8b0000; /* Dark Red */
         }
    </style>
""", unsafe_allow_html=True)

# --- App Content ---
st.title("📧 Spam Detector AI")
st.markdown("<p style='text-align: center; color: #555;'>Enter an email or SMS message below to check if it's likely spam or not.</p>", unsafe_allow_html=True)

# --- Example Messages ---
example_spam = "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."
example_ham = "Ok lar... Joking wif u oni..."

# Initialize session state for the text area if it doesn't exist
if 'input_mail' not in st.session_state:
    st.session_state.input_mail = ""

col1, col2 = st.columns(2)
with col1:
    if st.button("Load Example Spam"):
        st.session_state.input_mail = example_spam
with col2:
    if st.button("Load Example Ham"):
        st.session_state.input_mail = example_ham

# Display the text area - ONLY use key parameter, which auto-binds to session state
st.text_area(
    "Enter the message here:",
    height=150,
    placeholder="Type or paste your message...",
    key="input_mail"  # This automatically binds to st.session_state.input_mail
)

# --- Analysis Button ---
if st.button("Analyze Message"):
    input_mail_to_analyze = st.session_state.input_mail

    if input_mail_to_analyze:
        try:
            # --- Perform Text Analysis ---
            length, word_count, percent_upper, percent_digits, found_keywords = analyze_text(input_mail_to_analyze)

            # --- Prediction ---
            input_features = vectorizer.transform([input_mail_to_analyze])

            if input_features.nnz == 0:
                 st.warning("The message contains no analyzable words after basic filtering (stop words). Please enter more text.")
            else:
                prediction = model.predict(input_features)
                prediction_proba = model.predict_proba(input_features)
                spam_probability = prediction_proba[0][0]
                ham_probability = prediction_proba[0][1]

                # --- Display Results ---
                st.subheader("Analysis Result:")
                if prediction[0] == 1: # Ham
                    st.success(f"✅ Looks like **Ham** (Not Spam)")
                    st.progress(ham_probability)
                    st.markdown(f"<p style='text-align: center;'>Confidence: {ham_probability*100:.2f}%</p>", unsafe_allow_html=True)
                else: # Spam
                    st.error(f"🚨 Looks like **Spam**")
                    st.progress(spam_probability)
                    st.markdown(f"<p style='text-align: center;'>Confidence: {spam_probability*100:.2f}%</p>", unsafe_allow_html=True)

                # --- Display Text Analysis Features ---
                st.markdown("---") # Separator
                st.write("**Message Characteristics:**")
                # Use columns for better layout
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Length", f"{length} chars")
                col_b.metric("Words", f"{word_count}")
                col_c.metric("Uppercase", f"{percent_upper:.1f}%")
                col_d.metric("Digits", f"{percent_digits:.1f}%")

                if found_keywords:
                    st.write("**Potential Spam Keywords Found:**")
                    # Display keywords with some styling
                    keywords_html = "".join([f"<span class='keyword'>{kw}</span>" for kw in found_keywords])
                    st.markdown(keywords_html, unsafe_allow_html=True)
                else:
                     st.write("**Potential Spam Keywords Found:** None")

                # --- Details Expander ---
                with st.expander("Show Prediction Details"):
                    st.write(f"Probability of being Ham: {ham_probability*100:.2f}%")
                    st.write(f"Probability of being Spam: {spam_probability*100:.2f}%")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.error("Please ensure the input text is valid.")
    else:
        st.warning("Please enter a message to analyze.")

# --- Footer ---
st.markdown("<div class='footer'>Built with Streamlit and Scikit-learn</div>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a Machine Learning model (Logistic Regression trained on TF-IDF features) "
    "to classify messages as 'Spam' or 'Ham'. Basic preprocessing (lowercase, stop words) is handled by the TF-IDF vectorizer."
)
st.sidebar.markdown("---")
st.sidebar.write("**Model Performance (Test Set):**")
# --- Replace values below with YOUR results from the NOTEBOOK ---
st.sidebar.write("Model Used: Logistic Regression") # Or Naive Bayes if you saved that one
st.sidebar.write("Accuracy: 0.9668")  # <-- Replace with your Test Accuracy
st.sidebar.write("Spam F1-Score: 0.87") # <-- Replace with your Test F1-Score for Spam
st.sidebar.write("Ham F1-Score: 0.98") # <-- Replace with your Test F1-Score for Ham