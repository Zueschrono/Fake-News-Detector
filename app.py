import streamlit as st
import joblib

vectorizer = joblib.load('vectorizer.jb')
model = joblib.load('lr_model.jb')

st.set_page_config(page_title="📰 Fake News Detector", layout="centered")

st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stTextArea textarea {
            font-size: 16px;
        }
        .title {
            font-size: 100px;
            color: #2c3e50;
            font-weight: bold;
        }
        .subtitle {
            font-size: 20px;
            color: #7f8c8d;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🧠 Fake News Detection App</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter a news article below to find out whether it\'s real or fake.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("🛠 App Info")
    st.info("🔍 This app uses a Machine Learning model (Logistic Regression) trained to detect fake news articles.")
    st.markdown("**📂 Model Files:**")
    st.write("• `vectorizer.jb`\n• `lr_model.jb`")

news_input = st.text_area("📝 Paste your news article below:", height=200)

if st.button("🚀 Analyze News"):
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("✅ The News is **Real**!")
            st.balloons()
        else:
            st.error("❌ The News is **Fake**!")
            st.warning("Be cautious! Double-check the source.")
    else:
        st.warning("⚠️ Please enter some text before analyzing.")

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | [GitHub](https://github.com/)")
