from __future__ import annotations

import json, os, re, urllib.parse
from typing import List, Dict, Optional

import openai
import requests
from dotenv import load_dotenv

load_dotenv()
OPENAI_KEY   = os.getenv("OPENAI_API_KEY")
UNSPLASH_KEY = os.getenv("UNSPLASH_KEY")    

client = openai.OpenAI(api_key=OPENAI_KEY)

UA = {"User-Agent": "Mozilla/5.0", "Accept-Version": "v1"}
_PLACEHOLDER = "https://via.placeholder.com/120?text=No+Image"

def _amazon_search(keyword: str) -> str:
    return "https://www.amazon.com/s?k=" + urllib.parse.quote_plus(keyword)


def _first_amazon_dp(keyword: str) -> Optional[str]:
    """
    DuckDuckGo HTML → first /dp/ASIN link from amazon.com.
    """
    try:
        q = urllib.parse.quote_plus(f"{keyword} site:amazon.com/dp")
        html = requests.get(f"https://duckduckgo.com/html/?q={q}", headers=UA, timeout=5).text
        html = urllib.parse.unquote(html)
        m = re.search(r"https://www\.amazon\.com/[^\"]+/dp/[A-Z0-9]{10}", html)
        return m.group(0) if m else None
    except Exception:
        return None


def _is_amazon(url: str) -> bool:
    return url.lower().startswith("https://www.amazon.com")


def _unsplash_thumb(keyword: str) -> Optional[str]:
    if not UNSPLASH_KEY:
        return None
    try:
        r = requests.get(
            "https://api.unsplash.com/search/photos",
            params={"query": keyword, "per_page": 1, "orientation": "squarish"},
            headers={"Authorization": f"Client-ID {UNSPLASH_KEY}", **UA},
            timeout=4,
        )
        r.raise_for_status()
        hits = r.json().get("results", [])
        return hits[0]["urls"]["thumb"] if hits else None
    except Exception:
        return None


def _fix_recs(recs: List[Dict]) -> List[Dict]:
    out = []
    for r in recs:
        name = (r.get("name") or "item").strip()
        url  = (r.get("url")  or "").strip()

        if not _is_amazon(url):               
            dp_link = _first_amazon_dp(name)
            url = dp_link if dp_link else _amazon_search(name)

        r["url"] = url
        r["img"] = _unsplash_thumb(name) or _PLACEHOLDER
        out.append(r)
    return out


SYSTEM_PROMPT = (
    "You are a profiling assistant at an advertising company. Given Amazon order summaries (date, item, price, address),\n"
    "generate a JSON with two keys: 'profile' and 'recommendations'.\n"
    "- 'profile': a compact dict with high-level traits: age, gender, profession, lifestyle, personality, shopping style, hobbies, etc.\n"
    "- 'recommendations': list of 3–5 personalized suggestions (products, services, courses, etc.). Each has 'name', 'reason', 'url'.\n"
    "- 'IMPORTANT': Check that each url is valid. If not, replace it with a valid Amazon product or search link.\n"
    "Keep it short, focused, and avoid repeating the input."
)

def infer_user_profile(cleaned_prompt: str) -> Dict:
    response = client.chat.completions.create(
        model="o4-mini-2025-04-16",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": cleaned_prompt},
        ],
    )

    reply = response.choices[0].message.content.strip()

    if reply.startswith("```"):
        reply = re.sub(r"^```(\w+)?", "", reply).rstrip("```").strip()

    try:
        data: Dict = json.loads(reply)
    except json.JSONDecodeError:
        return {"raw_response": reply, "error": "GPT returned non-JSON"}

    if "recommendations" in data:
        data["recommendations"] = _fix_recs(data["recommendations"])

    return data
