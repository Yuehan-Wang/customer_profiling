import openai
import os
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def infer_user_profile(prompt_text, model="o4-mini-2025-04-16"):
    system_msg = (
        "You are an expert user profiler working for an ethical data analysis tool.\n"
        "Given a list of Amazon purchases (with date, product, price, and address),\n"
        "infer high-level characteristics of the user, such as:\n"
        "- age group (under 10, 10-19, 20-29, 30-39, 40-49, 50-59, 60+)\n"
        "- gender (if likely)\n"
        "- occupation or lifestyle\n"
        "- income level\n"
        "- educational background\n"
        "- relationship status\n"
        "- living situation (e.g. alone, with family, dormitory)\n"
        "- household role (e.g. primary buyer, caregiver, student)\n"
        "- shopping habits (impulsive, planned, bulk, brand-loyal, discount-seeker)\n"
        "- psychological traits (organized, anxious, pragmatic, indulgent, etc.)\n"
        "- values (sustainability, productivity, convenience, etc.)\n"
        "- hobbies and interests\n"
        "Summarize this as a JSON object with keys like 'age', 'gender', 'profession',\n"
        "'income', 'education', 'relationship', 'living', 'role', 'personality',\n"
        "'shopping_behavior', 'values', 'hobbies', etc.\n"
        "Keep it concise but insightful."
    )

    MAX_LINES = 500  
    print(len(prompt_text))
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
    try:
        result = eval(reply) if reply.strip().startswith("{") else {"raw": reply.strip()}
    except Exception:
        result = {"raw": reply.strip()}

    return result


if __name__ == "__main__":
    from data.data_cleaner import clean_order_csv

    text = clean_order_csv("data/Retail.OrderHistory.1.csv")
    profile = infer_user_profile(text)
    print(profile)
