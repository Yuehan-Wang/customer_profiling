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
import pandas as pd
import re

st.set_page_config(page_title="What Amazon Knows About You", layout="wide", page_icon="👤") 

COLOR_BACKGROUND = "#FFFBDE" 
COLOR_ACCENT_LIGHT = "#90D1CA" 
COLOR_TEXT_MEDIUM = "#129990" 
COLOR_TEXT_DARK = "#096B68"   
COLOR_WHITE = "#FFFFFF"
COLOR_DARK_FOR_LIGHT_ACCENT = "#000000" 
COLOR_MATCH_GREEN_BG = "#DFF0D8" 
COLOR_MATCH_GREEN_TEXT = "#3C763D" 

page_bg_style = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-color: {COLOR_BACKGROUND};
}}
[data-testid="stHeader"] {{
    background-color: {COLOR_BACKGROUND}; 
}}
h1, h2, h3, h4, h5, h6, p, li, label, [data-testid="stMarkdownContainer"] p {{
    color: {COLOR_TEXT_DARK};
}}
h1 {{ 
    color: {COLOR_TEXT_DARK};
    font-size: 2.5em; 
    text-align: center; 
    padding-top: 0.5em;
    padding-bottom: 0.5em;
}}
h2 {{ 
    color: {COLOR_TEXT_DARK};
    border-bottom: 2px solid {COLOR_TEXT_MEDIUM}; 
    padding-bottom: 0.3em;
    margin-top: 1.5em;
    margin-bottom: 0.8em;
}}
h3 {{ 
    color: {COLOR_TEXT_DARK}; 
    margin-top: 1.2em;
    margin-bottom: 0.5em;
}}
[data-testid="stSidebar"] > div:first-child {{
    background-color: #e8f5f3; 
    padding: 1em;
    border-right: 1px solid {COLOR_ACCENT_LIGHT};
}}
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, 
[data-testid="stSidebar"] div[role="listbox"] div, [data-testid="stSidebar"] div[data-baseweb="select"] > div {{
    color: {COLOR_TEXT_DARK} !important; 
}}
[data-testid="stSidebar"] .stButton button {{
    background-color: {COLOR_TEXT_MEDIUM};
    color: {COLOR_WHITE}; 
    border-radius: 5px;
    border: none;
    width: 100%; 
    padding: 0.5em;
    transition: background-color 0.3s ease, color 0.3s ease; 
}}
[data-testid="stSidebar"] .stButton button:hover {{
    background-color: {COLOR_TEXT_DARK};
    color: {COLOR_WHITE} !important; 
}}
[data-testid="stInfo"], [data-testid="stSuccess"], [data-testid="stError"], [data-testid="stWarning"] {{
    border-radius: 5px;
    padding: 1rem;
    color: {COLOR_TEXT_DARK}; 
}}
[data-testid="stInfo"] {{
    background-color: #e0f7fa; 
    border-left: 5px solid {COLOR_TEXT_MEDIUM};
}}
[data-testid="stSuccess"] {{ 
    background-color: {COLOR_MATCH_GREEN_BG}; 
    border-left: 5px solid {COLOR_MATCH_GREEN_TEXT};
    color: {COLOR_MATCH_GREEN_TEXT};
}}
[data-testid="stSuccess"] strong {{
    color: {COLOR_MATCH_GREEN_TEXT};
}}
.comparison-column {{
    padding: 15px;
    border-radius: 8px;
    background-color: {COLOR_WHITE}; 
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    margin-bottom: 1em; 
}}
.comparison-column h3 {{
    margin-top: 0; 
    color: {COLOR_TEXT_DARK};
}}
.profile-attribute {{
    margin-bottom: 10px; 
    line-height: 1.6;
}}
.profile-attribute strong {{
    color: {COLOR_TEXT_DARK};
}}
.profile-attribute em {{ 
    color: {COLOR_TEXT_MEDIUM};
}}
.recommendation-card {{
    display: flex;
    align-items: flex-start;
    border: 1px solid {COLOR_TEXT_MEDIUM}; 
    background-color: {COLOR_WHITE};
    border-radius: 8px;
    padding: 15px; 
    margin-bottom: 15px; 
    box-shadow: 0 2px 5px rgba(0,0,0,0.07);
}}
.recommendation-card img {{
    width: 90px; 
    height: 90px;
    object-fit: contain;
    margin-right: 15px;
    border-radius: 4px;
    border: 1px solid #eee; 
}}
.recommendation-card .content a {{
    font-size: 1.05em; 
    font-weight: bold;
    color: {COLOR_TEXT_DARK}; 
    text-decoration: none;
}}
.recommendation-card .content a:hover {{
    color: {COLOR_TEXT_MEDIUM}; 
    text-decoration: underline;
}}
.recommendation-card .content p {{
    margin-top: 5px;
    color: {COLOR_TEXT_DARK}; 
    font-size: 0.9em;
    line-height: 1.4;
}}
.footer-box {{
    background-color: {COLOR_WHITE}; 
    padding: 20px; 
    border-radius: 8px; 
    border: 1px solid {COLOR_TEXT_MEDIUM};
    margin-top: 2em;
}}
.footer-box p, .footer-box li {{
    color: {COLOR_TEXT_DARK};
}}
.footer-box strong {{
    color: {COLOR_TEXT_DARK};
}}

.stMultiSelect [data-baseweb="tag"] {{
    background-color: {COLOR_MATCH_GREEN_BG} !important;
    color: {COLOR_MATCH_GREEN_TEXT} !important;
    border: none !important;
    font-weight: bold;
    padding-left: 1em !important;
}}
.stMultiSelect [data-baseweb="tag"]:hover {{
    background-color: {COLOR_MATCH_GREEN_TEXT} !important;
    color: {COLOR_WHITE} !important;
}}
.stMultiSelect [data-baseweb="tag"] span {{
    color: inherit !important;
}}
.stMultiSelect [data-baseweb="tag"] svg {{
    color: {COLOR_MATCH_GREEN_TEXT} !important;
}}
.stMultiSelect [data-baseweb="tag"]:hover svg {{
    color: {COLOR_WHITE} !important;
}}
</style>
"""
st.markdown(page_bg_style, unsafe_allow_html=True)

st.title("What Amazon Knows About You")

if 'personal_info_saved' not in st.session_state:
    st.session_state.personal_info_saved = False
if 'self_profile_data' not in st.session_state:
    st.session_state.self_profile_data = {}

AGE_RANGES_USER = ["Prefer not to say", "Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
GENDER_OPTIONS_USER = ["Prefer not to say", "Male", "Female", "Non-binary", "Other"]
PROFESSION_OPTIONS_USER = ["Prefer not to say", "Student", "Employed", "Self-employed/Freelancer", "Unemployed", "Retired", "Homemaker", "Other"]
LIFESTYLE_OPTIONS_USER = ["Active", "Homebody", "Tech-focused", "Academic", "Social", "Outdoorsy", "Minimalist", "Family-oriented", "Health-conscious"]
PERSONALITY_OPTIONS_USER = ["Outgoing", "Introverted", "Practical", "Creative", "Analytical", "Spontaneous", "Organized", "Easy-going", "Detail-oriented"]
HOBBY_OPTIONS_USER = ["Reading", "Gaming", "Cooking", "Sports", "Traveling", "Music", "Movies/TV", "Art/Crafts", "Gardening", "Tech/Coding", "Fitness", "Writing"]
SHOPPING_STYLE_OPTIONS_USER = ["Budget-conscious", "Brand-loyal", "Impulse buyer", "Researcher", "Comfort-seeker", "Trend-follower", "Quality-focused", "Eco-conscious"]

st.sidebar.header("Your Self-Perception") 
st.sidebar.markdown("Tell us a bit about yourself.")

self_age_default_index = 0
if st.session_state.personal_info_saved and st.session_state.self_profile_data.get("age") in AGE_RANGES_USER:
    self_age_default_index = AGE_RANGES_USER.index(st.session_state.self_profile_data.get("age"))
self_age = st.sidebar.selectbox("Your Age Range", AGE_RANGES_USER, index=self_age_default_index)

self_gender_default_index = 0
if st.session_state.personal_info_saved and st.session_state.self_profile_data.get("gender") in GENDER_OPTIONS_USER:
    self_gender_default_index = GENDER_OPTIONS_USER.index(st.session_state.self_profile_data.get("gender"))
self_gender = st.sidebar.selectbox("Your Gender", GENDER_OPTIONS_USER, index=self_gender_default_index)

self_profession_default_index = 0
if st.session_state.personal_info_saved and st.session_state.self_profile_data.get("profession") in PROFESSION_OPTIONS_USER:
    self_profession_default_index = PROFESSION_OPTIONS_USER.index(st.session_state.self_profile_data.get("profession"))
self_profession = st.sidebar.selectbox("Your Profession/Student Status", PROFESSION_OPTIONS_USER, index=self_profession_default_index)

self_lifestyle_default = []
if st.session_state.personal_info_saved:
    self_lifestyle_default = [opt for opt in st.session_state.self_profile_data.get("lifestyle", []) if opt in LIFESTYLE_OPTIONS_USER]
self_lifestyle = st.sidebar.multiselect("Describe your Lifestyle", LIFESTYLE_OPTIONS_USER, default=self_lifestyle_default)

self_personality_default = []
if st.session_state.personal_info_saved:
    self_personality_default = [opt for opt in st.session_state.self_profile_data.get("personality", []) if opt in PERSONALITY_OPTIONS_USER]
self_personality = st.sidebar.multiselect("Describe your Personality", PERSONALITY_OPTIONS_USER, default=self_personality_default)

self_hobbies_default = []
if st.session_state.personal_info_saved:
    self_hobbies_default = [opt for opt in st.session_state.self_profile_data.get("hobbies", []) if opt in HOBBY_OPTIONS_USER]
self_hobbies = st.sidebar.multiselect("Your Hobbies", HOBBY_OPTIONS_USER, default=self_hobbies_default)

self_shopping_style_default = []
if st.session_state.personal_info_saved:
    self_shopping_style_default = [opt for opt in st.session_state.self_profile_data.get("shopping_style", []) if opt in SHOPPING_STYLE_OPTIONS_USER]
self_shopping_style = st.sidebar.multiselect("Your Shopping Style", SHOPPING_STYLE_OPTIONS_USER, default=self_shopping_style_default)

if st.sidebar.button("Save Personal Information"):
    st.session_state.self_profile_data = {
        "age": self_age if self_age != "Prefer not to say" else "",
        "gender": self_gender if self_gender != "Prefer not to say" else "",
        "profession": self_profession if self_profession != "Prefer not to say" else "",
        "lifestyle": self_lifestyle,
        "personality": self_personality,
        "hobbies": self_hobbies,
        "shopping_style": self_shopping_style
    }
    st.session_state.personal_info_saved = True
    st.sidebar.success("Personal information saved!") 
    st.rerun()

def get_comparison_display(key_name: str, self_val_raw, inferred_val_raw):
    match_style = f"background-color: {COLOR_MATCH_GREEN_BG}; color: {COLOR_MATCH_GREEN_TEXT}; padding: 2px 5px; border-radius: 4px; display: inline-block; font-weight: 500;"
    match_icon = "✅ " 
    
    self_val_display_processed = ""
    if not self_val_raw or (isinstance(self_val_raw, str) and self_val_raw == "Prefer not to say"):
        self_val_display_processed = "<em>Not specified</em>"
    elif isinstance(self_val_raw, list) and not self_val_raw:
        self_val_display_processed = "<em>Not specified</em>"
    else:
        self_val_display_processed = self_val_raw 

    if self_val_display_processed == "<em>Not specified</em>":
        return f"<div class='profile-attribute'><strong>{key_name.replace('_', ' ').capitalize()}:</strong> {self_val_display_processed}</div>"

    final_display_html_parts = []

    if isinstance(self_val_raw, list): 
        inferred_list_lower = []
        if isinstance(inferred_val_raw, list):
            inferred_list_lower = [str(item).lower() for item in inferred_val_raw]
        elif isinstance(inferred_val_raw, str): 
            inferred_list_lower = [item.strip().lower() for item in inferred_val_raw.split(',')]

        for item in self_val_raw:
            item_str = str(item)
            if item_str.lower() in inferred_list_lower:
                final_display_html_parts.append(f"<span style='{match_style}'>{match_icon}{item_str}</span>") 
            else:
                final_display_html_parts.append(item_str)
        self_val_display_processed = ", ".join(final_display_html_parts) if final_display_html_parts else "<em>Not specified</em>"
    
    else: 
        self_str_lower = str(self_val_raw).lower()
        inferred_str_lower = str(inferred_val_raw).lower()
        is_match = False

        if self_str_lower and inferred_str_lower and self_str_lower != "prefer not to say":
            if self_str_lower == inferred_str_lower: 
                is_match = True
            elif key_name == "profession" and self_str_lower in inferred_str_lower : 
                 is_match = True
            elif key_name == "age": 
                 if self_str_lower in inferred_str_lower: is_match = True

        if is_match:
            self_val_display_processed = f"<span style='{match_style}'>{match_icon}{str(self_val_raw)}</span>" 
        else:
            self_val_display_processed = str(self_val_raw)
            
    return f"<div class='profile-attribute'><strong>{key_name.replace('_', ' ').capitalize()}:</strong> {self_val_display_processed}</div>"

if not st.session_state.personal_info_saved:
    st.info("Please fill out and save your self-perception details in the sidebar to proceed.")
else:
    st.markdown("""
    This tool analyzes your uploaded Amazon Order CSV file to infer a potential user profile. 
    You can then compare this inferred profile with your own self-perception.
    This helps illustrate how companies might build customer profiles based on purchase data.
    """)
    uploaded_file = st.file_uploader("Upload your Amazon Order CSV file:", type=["csv"])

    if uploaded_file:
        st.markdown("<div class='main-content-area'>", unsafe_allow_html=True) 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        st.info("Processing your Amazon order history...")
        try:
            prompt_text = clean_order_csv(tmp_path) 
            os.remove(tmp_path)

            st.success(f"Processed {len(prompt_text.splitlines())} relevant order lines from your file.")
            st.caption(f"Input text length for analysis: {len(prompt_text)} characters")

            if not prompt_text.strip():
                st.error("The processed CSV file resulted in no data to analyze.")
            else:
                with st.spinner("Analyzing data and inferring profile..."):
                    profile_data = infer_user_profile(prompt_text) 

                st.header("Profile Comparison")
                
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown('<div class="comparison-column">', unsafe_allow_html=True)
                    st.subheader("Your Self-Perception")
                    self_data_to_display = st.session_state.self_profile_data
                    inferred_profile_map = profile_data.get("profile", {})

                    if any(val for val in self_data_to_display.values() if val): 
                        for key, self_value in self_data_to_display.items():
                            inferred_value = inferred_profile_map.get(key) 
                            comparison_html = get_comparison_display(key, self_value, inferred_value)
                            st.markdown(comparison_html, unsafe_allow_html=True)
                    else:
                        st.markdown("No self-perception data entered or all fields were 'Prefer not to say'")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="comparison-column">', unsafe_allow_html=True)
                    st.subheader("Profile Inferred from Your Data") 
                    
                    inferred_profile = profile_data.get("profile")
                    if inferred_profile:
                        for key, value in inferred_profile.items():
                            display_value_inferred = ""
                            if isinstance(value, list):
                                display_value_inferred = ", ".join(str(v) for v in value) if value else "<em>Not specified by AI</em>"
                            else:
                                display_value_inferred = str(value) if value else "<em>Not specified by AI</em>"
                            
                            st.markdown(f"<div class='profile-attribute'><strong>{key.capitalize()}:</strong> {display_value_inferred}</div>", unsafe_allow_html=True)
                    elif "error" in profile_data:
                         st.error(f"Could not display inferred profile. Error: {profile_data.get('error')}")
                         if profile_data.get("raw_response"):
                            st.expander("Raw GPT Response").code(profile_data.get("raw_response", ""), language="text")
                    else:
                        st.warning("No profile data was inferred, or the profile is empty.")
                    st.markdown('</div>', unsafe_allow_html=True)

                if "profile" in profile_data and profile_data["profile"]:
                    stopwords_set = set(STOPWORDS)
                    custom_stops = {
                        "likely", "no", "yes", "unknown", "prefer not to say", "not specified", 
                        "na", "n a", "etc", "one", "two", "also", "may", "might", "often", "usually",
                        "age", "gender", "profession", "lifestyle", "personality", "hobbies", "shopping_style",
                        "style", "focused", "oriented", "conscious", "loyal", "buyer", "seeker", "follower", "edition"
                    }
                    stopwords_set.update(custom_stops)
                    
                    keywords = []
                    for v_key, v_val in profile_data["profile"].items():
                        values_to_process = []
                        if isinstance(v_val, str):
                            if v_val.lower() not in custom_stops and v_val:
                                values_to_process.append(v_val)
                        elif isinstance(v_val, list):
                            for item_in_list in v_val:
                                if isinstance(item_in_list, str) and item_in_list.lower() not in custom_stops and item_in_list:
                                    values_to_process.append(item_in_list)
                        
                        for text_item in values_to_process:
                            text_item_cleaned = text_item.lower().translate(str.maketrans("", "", string.punctuation))
                            words = text_item_cleaned.split()
                            for word in words:
                                if word not in stopwords_set and len(word) > 2:
                                    keywords.append(word)
                    
                    if keywords:
                        st.header("Word Cloud from Inferred Profile Data") 
                        try:
                            tag_weights = dict(Counter(keywords))
                            if not tag_weights:
                                 st.caption("Not enough descriptive keywords to generate a word cloud.")
                            else: 
                                wc = WordCloud(width=800, height=400, background_color=COLOR_WHITE, 
                                               stopwords=stopwords_set, collocations=True, 
                                               colormap="viridis", min_font_size=10) 
                                wc.generate_from_frequencies(tag_weights)
                                fig, ax = plt.subplots()
                                ax.imshow(wc, interpolation='bilinear')
                                ax.axis("off")
                                fig.patch.set_facecolor(COLOR_BACKGROUND) 
                                st.pyplot(fig)
                        except Exception as e:
                            st.caption(f"Could not generate word cloud: {e}")
                    else:
                        st.caption("Not enough descriptive keywords for a word cloud (after filtering common/category terms).")

                if "recommendations" in profile_data and profile_data["recommendations"]:
                    st.header("Personalized Recommendations") 
                    st.markdown("<em>Based on the inferred profile, here are some product suggestions.</em>")

                    for rec in profile_data["recommendations"]:
                        name = rec.get("name", "N/A")
                        reason = rec.get("reason", "No reason provided.")
                        url = rec.get("url", "#")
                        img_url = rec.get("img") or "https://via.placeholder.com/120?text=No+Image+Available"

                        st.markdown(f"""
                        <div class="recommendation-card">
                            <img src="{img_url}" alt="{name}">
                            <div class="content">
                                <a href="{url}" target="_blank">{name}</a>
                                <p>{reason}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                if profile_data.get("profile"): 
                    st.info("Full JSON response from the profiling model:")
                    st.expander("Click to expand full JSON").code(
                        json.dumps(profile_data, indent=2, ensure_ascii=False), language="json"
                    )
        except ValueError as e:
            st.error(f"Error processing CSV: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path): 
                 os.remove(tmp_path) 
        st.markdown("</div>", unsafe_allow_html=True) 
    