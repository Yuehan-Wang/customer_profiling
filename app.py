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

    st.write("Processing CSV file...")
    prompt_text = clean_order_csv(tmp_path)
    os.remove(tmp_path)

    st.write(f"Processed {len(prompt_text.splitlines())} lines")
    st.write(f"Input text length: {len(prompt_text)} characters")

    with st.spinner("Calling GPT to infer profile..."):
        profile_data = infer_user_profile(prompt_text)

    st.subheader("GPT Response")
    st.expander("Click to expand full JSON").code(
        json.dumps(profile_data, indent=2, ensure_ascii=False), language="json"
    )

    if "profile" in profile_data:
        st.subheader("Inferred User Profile")
        for key, value in profile_data["profile"].items():
            st.markdown(f"<div style='margin-bottom: 8px;'><b>{key.capitalize()}:</b> {value}</div>", unsafe_allow_html=True)

        stopwords = set(STOPWORDS)
        custom_stops = {"likely", "no", "yes", "(", ")", "/", "&", "to", "the", "with", "and", "or", "of"}
        stopwords.update(custom_stops)

        keywords = []
        for v in profile_data["profile"].values():
            if isinstance(v, str):
                v = v.lower().translate(str.maketrans("", "", string.punctuation))
                keywords.extend([t for t in v.split() if t not in stopwords and len(t) > 2])

        if keywords:
            st.subheader("Word Cloud View")
            tag_weights = dict(Counter(keywords))
            wc = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords)
            wc.generate_from_frequencies(tag_weights)
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

        if "recommendations" in profile_data:
            st.subheader("Personalized Recommendations")

            for rec in profile_data["recommendations"]:
                name = rec.get("name", "")
                reason = rec.get("reason", "")
                url = rec.get("url", "")
                asin = url.split("/dp/")[-1].split("/")[0] if "/dp/" in url else ""
                img_url = f"https://images-na.ssl-images-amazon.com/images/P/{asin}.jpg" if asin else "https://via.placeholder.com/100"

                card_html = f"""
                <div style="display: flex; align-items: flex-start; border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                    <img src="{img_url}" alt="{name}" style="width: 100px; height: 100px; object-fit: contain; margin-right: 16px;">
                    <div>
                        <a href="{url}" target="_blank" style="font-size: 16px; font-weight: bold; color: #0066cc; text-decoration: none;">{name}</a>
                        <p style="margin-top: 4px; color: #444;">{reason}</p>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

