import openai
import os
from dotenv import load_dotenv
import logging
import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gpt_infer.log')  
    ]
)

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def infer_user_profile(prompt_text, model="o4-mini-2025-04-16"):
    system_msg = (
        "You are a profiling assistant at an advertising company. Given Amazon order summaries (date, item, price, address),\n"
        "generate a JSON with two keys: 'profile' and 'recommendations'.\n"
        "- 'profile': a compact dict with high-level traits: age, gender, profession, lifestyle, personality, shopping style, hobbies, etc.\n"
        "- 'recommendations': list of 3–5 personalized suggestions (products, services, courses, etc.). Each has 'name', 'reason', 'url'.\n"
        "- 'IMPORTANT': Check if the url is valid. If not, select a valid url from the recommendation.\n"
        "Keep it short, focused, and avoid repeating the input."
    )

    MAX_LINES = 500  
    logging.info(f"Input text length: {len(prompt_text)}")
    st.write(f"Input text length: {len(prompt_text)}")
    truncated = "\n".join(prompt_text.split("\n")[:MAX_LINES])

    user_msg = f"Here is the user's Amazon purchase history:\n{truncated}\n\nNow generate the high-level profile."

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    )

    reply = response.choices[0].message.content
    logging.info(f"GPT Response: {reply}")
    st.write("GPT:")
    st.code(reply, language="json")
    
    try:
        result = eval(reply) if reply.strip().startswith("{") else {"raw": reply.strip()}
        logging.info(f"Processed result: {result}")
        st.write("Processed result:")
        st.json(result)
    except Exception as e:
        logging.error(f"Error processing reply: {str(e)}")
        st.error(f"Error processing reply: {str(e)}")
        result = {"raw": reply.strip()}

    return result


if __name__ == "__main__":
    from data.data_cleaner import clean_order_csv

    text = clean_order_csv("data/Retail.OrderHistory.1.csv")
    profile = infer_user_profile(text)
    logging.info(f"Final profile: {profile}")
    st.write("Final profile:")
    st.json(profile)