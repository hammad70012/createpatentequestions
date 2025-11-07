import os, json, time, requests, random
from tqdm import tqdm
from dotenv import load_dotenv

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("OPENROUTER_KEY")
MODEL = "deepseek/deepseek-chat-v3-0324:free"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "quizPatenteB2023.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "data", "questions_multi_explained_batch.json")

URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def flatten_questions(data):
    """Flatten nested Italian quiz structure -> flat list."""
    flat = []
    counter = 1
    for section, categories in data.items():
        if isinstance(categories, dict):
            for category, questions in categories.items():
                if isinstance(questions, list):
                    for q in questions:
                        flat.append({
                            "id": counter,
                            "section": section,
                            "category": category,
                            "img": q.get("img", ""),
                            "question": q.get("q", ""),
                            "answer": q.get("a", None)
                        })
                        counter += 1
    print(f"✅ Flattened {len(flat)} questions.")
    return flat


def ask_openrouter_batch(batch, max_retries=6):
    """
    Send up to 5–10 questions to OpenRouter for translation + explanations.
    Includes exponential backoff and retry logic.
    """
    prompt = (
        "You are a multilingual professional driving theory teacher. "
        "Translate and explain each of the following Italian driving theory questions. "
        "For each question:\n"
        "- Translate the question into English, Urdu, and Hindi.\n"
        "- Provide a short explanation (1–2 lines) about what the question means.\n"
        "- Then, explain WHY the given answer (True or False) is correct or wrong in each language separately.\n\n"
        "Return ONLY a valid JSON array in this exact structure:\n\n"
        "[\n"
        "  {\n"
        "    \"id\": <id>,\n"
        "    \"it\": {\"question\": \"...\", \"why\": \"...\"},\n"
        "    \"en\": {\"question\": \"...\", \"explanation\": \"...\", \"why\": \"...\"},\n"
        "    \"ur\": {\"question\": \"...\", \"explanation\": \"...\", \"why\": \"...\"},\n"
        "    \"hi\": {\"question\": \"...\", \"explanation\": \"...\", \"why\": \"...\"}\n"
        "  }\n"
        "]\n\n"
        "Make sure each language includes its own 'why' field describing why the answer is right or wrong.\n"
        "Questions:\n"
    )

    for q in batch:
        prompt += f"\nID {q['id']}: {q['question']} (Answer: {'True' if q['answer'] else 'False'})"

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4
    }

    retry_delay = 10  # start with 10s
    for attempt in range(max_retries):
        try:
            r = requests.post(URL, headers=HEADERS, json=payload, timeout=120)
            if r.status_code == 429:
                # rate limit
                print(f"⏳ Rate limited (HTTP 429). Waiting {retry_delay}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"].strip()

            start = content.find("[")
            end = content.rfind("]") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found in response.")
            snippet = content[start:end]
            return json.loads(snippet)

        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout. Waiting {retry_delay}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(retry_delay)
            retry_delay *= 2
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Request error: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay *= 2
        except Exception as e:
            print(f"⚠️ Parse error: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay *= 2

    print("❌ All retries failed for this batch.")
    return None


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ------------------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------------------
def main():
    ensure_dir(OUTPUT_FILE)

    # Load dataset
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        questions = flatten_questions(data)
    elif isinstance(data, list):
        questions = data
    else:
        raise ValueError("Unexpected JSON structure")

    translated = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                translated = json.load(f)
            except Exception:
                translated = []

    done_ids = {q["id"] for q in translated}
    remaining = [q for q in questions if q["id"] not in done_ids]

    print(f"🔹 Total: {len(questions)} | Remaining: {len(remaining)}")

    batch_size = 10
    for i in tqdm(range(0, len(remaining), batch_size), desc="Batch Translating"):
        batch = remaining[i:i + batch_size]
        tr_list = ask_openrouter_batch(batch)

        if not tr_list:
            print("⚠️ Skipping batch due to repeated failure.")
            continue

        for tr in tr_list:
            match = next((q for q in batch if q["id"] == tr["id"]), None)
            if match:
                translated.append({
                    "id": match["id"],
                    "section": match.get("section"),
                    "category": match.get("category"),
                    "img": match.get("img"),
                    "question": {
                        "it": match.get("question", ""),
                        "en": tr["en"]["question"],
                        "ur": tr["ur"]["question"],
                        "hi": tr["hi"]["question"]
                    },
                    "answer": match.get("answer"),
                    "explanation": {
                        "en": tr["en"]["explanation"],
                        "ur": tr["ur"]["explanation"],
                        "hi": tr["hi"]["explanation"]
                    },
                    "why": {
                        "it": tr["it"]["why"],
                        "en": tr["en"]["why"],
                        "ur": tr["ur"]["why"],
                        "hi": tr["hi"]["why"]
                    }
                })

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(translated, f, ensure_ascii=False, indent=2)

        # random pause between batches to avoid rate limits
        time.sleep(random.uniform(3.0, 7.0))

    print(f"✅ All done! Translated + explained {len(translated)} questions → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
