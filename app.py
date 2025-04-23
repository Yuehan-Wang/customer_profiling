import streamlit as st
from gpt_infer import infer_user_profile
from data.data_cleaner import clean_order_csv
import tempfile
import os
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import string

st.set_page_config(page_title="User Profiler", layout="wide")
st.title("User Profile Inference from Amazon Orders")

uploaded_file = st.file_uploader("Upload your Amazon Order CSV file:", type=["csv"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    prompt_text = clean_order_csv(tmp_path)
    os.remove(tmp_path)

    cache_path = "last_profile.json"
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
    else:
        with st.spinner("Calling GPT to infer profile..."):
            profile = infer_user_profile(prompt_text)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

    st.subheader("Inferred Profile")

    if "raw" in profile:
        st.text(profile["raw"])
    else:
        st.json(profile)

        stopwords = set(STOPWORDS)
        custom_stops = {"likely", "no", "yes", "(", ")", "/", "&", "to", "the", "with", "and", "or", "of"}
        stopwords.update(custom_stops)

        keywords = []
        for v in profile.values():
            if isinstance(v, str):
                v = v.lower()
                v = v.translate(str.maketrans("", "", string.punctuation)) 
                tokens = v.split()
                tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
                keywords.extend(tokens)

        tag_weights = dict(Counter(keywords))

        if tag_weights:
            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                stopwords=stopwords
            ).generate_from_frequencies(tag_weights)

            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.subheader("Word Cloud View")
            st.pyplot(fig)
