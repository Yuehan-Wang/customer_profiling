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
_PLACEHOLDER = "https://via.placeholder.com/120?text=No+Image+Available"

AGE_RANGES = ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+", "Unknown"]
GENDER_OPTIONS = ["Male", "Female", "Non-binary", "Other", "Unknown"]
PROFESSION_OPTIONS = ["Student", "Employed", "Self-employed/Freelancer", "Unemployed", "Retired", "Homemaker", "Other", "Unknown"]
LIFESTYLE_OPTIONS = ["Active", "Homebody", "Tech-focused", "Academic", "Social", "Outdoorsy", "Minimalist", "Family-oriented", "Health-conscious"]
PERSONALITY_OPTIONS = ["Outgoing", "Introverted", "Practical", "Creative", "Analytical", "Spontaneous", "Organized", "Easy-going", "Detail-oriented"]
HOBBY_OPTIONS = ["Reading", "Gaming", "Cooking", "Sports", "Traveling", "Music", "Movies/TV", "Art/Crafts", "Gardening", "Tech/Coding", "Fitness", "Writing"]
SHOPPING_STYLE_OPTIONS = ["Budget-conscious", "Brand-loyal", "Impulse buyer", "Researcher", "Comfort-seeker", "Trend-follower", "Quality-focused", "Eco-conscious"]

SYSTEM_PROMPT = f"""
Generate a JSON output with 'profile' and 'recommendations' keys from Amazon order summaries.
'profile': Compact dict. For traits below, STRICTLY select from the given options.
- 'age' (ONE from): {AGE_RANGES}
- 'gender' (ONE from): {GENDER_OPTIONS}
- 'profession' (ONE from): {PROFESSION_OPTIONS}
- 'lifestyle' (LIST of one or more from): {LIFESTYLE_OPTIONS}
- 'personality' (LIST of one or more from): {PERSONALITY_OPTIONS}
- 'hobbies' (LIST of one or more from): {HOBBY_OPTIONS}
- 'shopping_style' (LIST of one or more from): {SHOPPING_STYLE_OPTIONS}
If undetermined, use "Unknown" for single-choice; omit multi-choice if none apply.

'recommendations': List of 3-5 suggested items (each a dict with 'name', 'reason', 'url').
- 'name': Provide an ACCURATE, concise, and SPECIFIC product title that clearly describes the item. For example, instead of just 'USB Hub', use 'Anker USB C Hub, 5-in-1 Adapter'. This title will be used for generating fallback search URLs AND for finding a relevant image. A descriptive name like 'Brand Model Type of Product' is key for good image matching.
- 'url': CRITICAL - Provide a direct, VALID, and WORKING Amazon.com product page URL (must contain '/dp/ASIN', e.g., https://www.amazon.com/dp/B01F8XCDHI). Verify the ASIN leads to an active product page.
    - If, and ONLY IF, a direct product page URL cannot be confidently provided, THEN generate a specific Amazon.com search URL (e.g., https://www.amazon.com/s?k=exact+product+name) that is highly likely to show the intended product as a top result.
    - DO NOT provide URLs to non-Amazon sites. DO NOT invent ASINs or use placeholder ASINs. All URLs must be for amazon.com.

Response MUST be compact JSON. Do not repeat input summaries.
Profile example: {{ "age": "25-34", "gender": "Female", "profession": "Employed", "lifestyle": ["Active"], "hobbies": ["Gaming", "Reading"], "shopping_style": ["Researcher"] }}
"""

def _amazon_search(keyword: str) -> str:
    if not keyword: 
        return "https://www.amazon.com/" 
    return "https://www.amazon.com/s?k=" + urllib.parse.quote_plus(keyword)


def _first_amazon_dp(keyword: str) -> Optional[str]:
    if not keyword:
        return None
    try:
        q = urllib.parse.quote_plus(f"{keyword} site:amazon.com/dp")
        search_url = f"https://html.duckduckgo.com/html/?q={q}"
        
        html = requests.get(search_url, headers=UA, timeout=7).text
        html = urllib.parse.unquote(html)
        
        match = re.search(r"https://www\.amazon\.com/(?:[^\"\'\s]+/)?(?:dp|gp/product)/([A-Z0-9]{10})", html, re.IGNORECASE)
        
        if match:
            asin = match.group(1)
            return f"https://www.amazon.com/dp/{asin}"
        return None
    except requests.exceptions.RequestException:
        return None
    except Exception: 
        return None


def _is_amazon_product_page(url: str) -> bool:
    if not isinstance(url, str): return False
    return bool(re.match(r"https://www\.amazon\.com/(?:[^/\s]+/)?(?:dp|gp/product)/[A-Z0-9]{10}", url, re.IGNORECASE))

def _is_amazon_domain(url: str) -> bool:
    if not isinstance(url, str): return False
    return url.lower().startswith("https://www.amazon.com")


def _unsplash_thumb(keyword: str, page: int = 1) -> Optional[str]: 
    if not UNSPLASH_KEY or not keyword:
        return None
    
    search_query = keyword 

    try:
        r = requests.get(
            "https://api.unsplash.com/search/photos",
            params={"query": search_query, "per_page": 1, "orientation": "squarish", "page": page},
            headers={"Authorization": f"Client-ID {UNSPLASH_KEY}", **UA},
            timeout=4,
        )
        r.raise_for_status() 
        hits = r.json().get("results", [])
        return hits[0]["urls"]["thumb"] if hits else None
    except requests.exceptions.RequestException: 
        return None 
    except (KeyError, IndexError, ValueError): 
        return None
    except Exception:
        return None


def _fix_recs(recs: List[Dict]) -> List[Dict]:
    out = []
    if not isinstance(recs, list):
        return out
    
    used_image_urls = set()

    for r_item in recs:
        if not isinstance(r_item, dict):
            continue

        name = (r_item.get("name") or "").strip()
        original_url_from_gpt  = (r_item.get("url")  or "").strip()
        current_url = original_url_from_gpt

        if not name:
            r_item["url"] = current_url or "https://www.amazon.com"
            r_item["img"] = _PLACEHOLDER
            out.append(r_item)
            continue

        if not _is_amazon_product_page(current_url):
            dp_link = _first_amazon_dp(name) 
            if dp_link:
                current_url = dp_link
            elif _is_amazon_domain(original_url_from_gpt) and original_url_from_gpt: 
                current_url = original_url_from_gpt 
            else: 
                current_url = _amazon_search(name)
        
        if not current_url:
            current_url = _amazon_search(name)
        
        r_item["url"] = current_url

        img_url = _unsplash_thumb(name, page=1)
        if img_url and img_url in used_image_urls: 
            img_url_page2 = _unsplash_thumb(name, page=2) 
            if img_url_page2 and img_url_page2 not in used_image_urls:
                img_url = img_url_page2
            else: 
                img_url = _PLACEHOLDER 
        
        if img_url and img_url != _PLACEHOLDER:
            used_image_urls.add(img_url)
            r_item["img"] = img_url
        else:
            r_item["img"] = _PLACEHOLDER 

        out.append(r_item)
    return out


def infer_user_profile(cleaned_prompt: str) -> Dict:
    try:
        response = client.chat.completions.create(
            model="o4-mini-2025-04-16", 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": cleaned_prompt},
            ],
            response_format={"type": "json_object"},
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        return {"raw_response": "", "error": f"OpenAI API call failed: {str(e)}", "profile": {}, "recommendations": []}
    
    data: Dict = {}
    try:
        data = json.loads(reply)
    except json.JSONDecodeError:
        match = re.search(r"```json\s*([\s\S]*?)\s*```", reply, re.MULTILINE)
        if match:
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                 return {"raw_response": reply, "error": "GPT returned non-JSON content even after markdown extraction.", "profile": {}, "recommendations": []}
        else:
            return {"raw_response": reply, "error": "GPT returned non-JSON content.", "profile": {}, "recommendations": []}

    if "profile" not in data or not isinstance(data.get("profile"), dict):
        data["profile"] = {}

    profile_schema = {
        "age": {"type": "string", "options": AGE_RANGES},
        "gender": {"type": "string", "options": GENDER_OPTIONS},
        "profession": {"type": "string", "options": PROFESSION_OPTIONS},
        "lifestyle": {"type": "list", "options": LIFESTYLE_OPTIONS},
        "personality": {"type": "list", "options": PERSONALITY_OPTIONS},
        "hobbies": {"type": "list", "options": HOBBY_OPTIONS},
        "shopping_style": {"type": "list", "options": SHOPPING_STYLE_OPTIONS},
    }

    cleaned_profile = {}
    profile_from_gpt = data.get("profile", {})
    for key, schema_info in profile_schema.items():
        value = profile_from_gpt.get(key)
        if schema_info["type"] == "string":
            if isinstance(value, str) and value in schema_info["options"]:
                cleaned_profile[key] = value
            else: 
                cleaned_profile[key] = next((opt for opt in schema_info["options"] if opt == "Unknown"), schema_info["options"][-1])
        elif schema_info["type"] == "list":
            if isinstance(value, list):
                cleaned_profile[key] = [str(v) for v in value if isinstance(v, str) and v in schema_info["options"]]
            elif isinstance(value, str): 
                 cleaned_profile[key] = [v.strip() for v in value.split(',') if v.strip() in schema_info["options"]]
            else: 
                cleaned_profile[key] = []
    data["profile"] = cleaned_profile

    if "recommendations" in data and isinstance(data["recommendations"], list):
        data["recommendations"] = _fix_recs(data["recommendations"])
    else: 
        data["recommendations"] = _fix_recs(data.get("recommendations", []))

    return data