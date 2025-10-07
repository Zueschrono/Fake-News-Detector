import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Try importing TextBlob safely
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ModuleNotFoundError:
    TEXTBLOB_AVAILABLE = False

# Load model and vectorizer
vectorizer = joblib.load('vectorizer.jb')
model = joblib.load('rf_model.jb')

# Streamlit configuration
st.set_page_config(page_title="📰 Fake News Detector", layout="centered")

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stTextArea textarea {
            font-size: 16px;
        }
        .title {
            font-size: 120px;
            color: #2c3e50;
            font-weight: bold;
            text-align: center;
            margin-top: -30px;
        }
        .subtitle {
            font-size: 22px;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 30px;
        }
        .insight {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<p class="title">🧠 Fake News Detection App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter a news article below to find out whether it\'s real or fake.</p>', unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.header("🛠 App Info")
    st.info("🔍 This app uses a Machine Learning model (Random Forest) trained to detect fake news articles.")
    st.markdown("**📂 Model Files:**")
    st.write("• `vectorizer.jb`\n• `rf_model.jb`")

# Text input
news_input = st.text_area("📝 Paste your news article below:", height=200)

# Analysis button
if st.button("🚀 Analyze News"):
    if news_input.strip():
        # Transform input and predict
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)
        probability = model.predict_proba(transform_input)[0]

        # Determine result
        label = "Real" if prediction[0] == 1 else "Fake"
        confidence = probability[1] if prediction[0] == 1 else probability[0]

        # Display prediction
        if label == "Real":
            st.success(f"✅ The News is **Real**! (Confidence: {confidence:.2f})")
            st.balloons()
        else:
            st.error(f"❌ The News is **Fake**! (Confidence: {confidence:.2f})")
            st.warning("Be cautious! Double-check the source.")

        # --- DYNAMIC INSIGHTS SECTION ---
        word_count = len(news_input.split())
        sentence_count = news_input.count('.') + news_input.count('!') + news_input.count('?')

        # Sentiment analysis
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(news_input)
            sentiment_score = blob.sentiment.polarity
            if sentiment_score > 0.1:
                sentiment = "😊 Positive"
            elif sentiment_score < -0.1:
                sentiment = "😠 Negative"
            else:
                sentiment = "😐 Neutral"
        else:
            sentiment = "😐 Neutral (TextBlob not installed)"

        # Common words
        words = [w.lower() for w in news_input.split() if len(w) > 4]
        common_words = Counter(words).most_common(5)
        top_words = ", ".join([w for w, _ in common_words]) if common_words else "N/A"

        # Build statistics table
        stats = pd.DataFrame({
            "Metric": [
                "Word Count",
                "Sentence Count",
                "Sentiment",
                "Most Frequent Words",
                "Model Confidence",
                "Predicted Label"
            ],
            "Value": [
                word_count,
                sentence_count,
                sentiment,
                top_words,
                f"{confidence:.2f}",
                label
            ]
        })

        # Display table
        st.markdown("---")
        st.markdown('<div class="insight"><h3>📊 Article Statistics</h3></div>', unsafe_allow_html=True)
        st.dataframe(stats, use_container_width=True)

        # --- STATISTICAL GRAPH (BAR CHART) ---
        st.markdown("<h4 style='text-align:center;'>📈 Model Prediction Probability</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.bar(["Fake", "Real"], probability, color=["#e74c3c", "#2ecc71"])
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.set_title("Prediction Confidence Levels")
        for i, v in enumerate(probability):
            ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)
        st.pyplot(fig)

        st.markdown("<p style='color:#7f8c8d;'>This graph shows the model's confidence for both classes.</p>", unsafe_allow_html=True)
