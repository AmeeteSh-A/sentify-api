import os
import re
import random
import html
import joblib
import uvicorn
import nltk
import time
import cloudscraper

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bs4 import BeautifulSoup
from googlesearch import search
from unidecode import unidecode
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import lyricsgenius

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =================================================================================
# --- 1. CONFIGURATION & STARTUP ---
# =================================================================================

app = FastAPI(
    title="SpotifyMood API",
    description="API to find lyrics and predict song mood.",
    version="2.0.0"
)

GENIUS_API_TOKEN = "lyN7oYYT8zEG_I1eg1ySpesW_1wW9HnYg4Ctv6tZgw-wYoAncBsXYs-sWJZ1l3ES"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.90 Safari/537.36",
]

# Global objects
model = None
vectorizer = None
genius_client = None
scraper = None

# =================================================================================
# --- 2. STARTUP EVENT ---
# =================================================================================

@app.on_event("startup")
def load_models_and_clients():
    global model, vectorizer, genius_client, scraper

    print("Initializing startup...")

    # Initialize cloudscraper
    try:
        scraper = cloudscraper.create_scraper()
        print("✅ Cloudscraper initialized.")
    except Exception as e:
        print(f"⚠️ Cloudscraper failed: {e}")

    # Download NLTK data safely
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    try:
        WordNetLemmatizer().lemmatize("test")
    except LookupError:
        nltk.download('wordnet')

    # Load ML model/vectorizer safely
    try:
        print("Loading ML model and vectorizer...")
        model = joblib.load('mood_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        print("✅ ML components loaded.")
    except FileNotFoundError as e:
        print(f"⚠️ Model/vectorizer file not found: {e}")
        model = None
        vectorizer = None
    except Exception as e:
        print(f"⚠️ Failed to load ML components: {e}")
        model = None
        vectorizer = None

    # Initialize Genius API client safely
    if not GENIUS_API_TOKEN or "PASTE" in GENIUS_API_TOKEN:
        print("⚠️ Genius API token not set. Genius client disabled.")
        genius_client = None
    else:
        try:
            genius_client = lyricsgenius.Genius(
                GENIUS_API_TOKEN, verbose=False, remove_section_headers=True
            )
            print("✅ Genius client initialized.")
        except Exception as e:
            print(f"⚠️ Genius client failed to initialize: {e}")
            genius_client = None

# =================================================================================
# --- 3. HELPERS ---
# =================================================================================

def get_random_headers():
    return {"User-Agent": random.choice(USER_AGENTS)}

def safe_get(url, retries=3):
    for attempt in range(retries):
        try:
            resp = scraper.get(url, headers=get_random_headers(), timeout=10)
            if resp.status_code == 200 and "captcha" not in resp.text.lower():
                return resp
        except Exception as e:
            print(f"Request failed on attempt {attempt+1}: {e}")
        time.sleep(2 ** attempt)
    return None

def transliterate_if_needed(text):
    scripts_to_check = {
        "Devanagari": (r'[\u0900-\u097F]', sanscript.DEVANAGARI),
        "Gurmukhi":   (r'[\u0A00-\u0A7F]', sanscript.GURMUKHI),
    }
    for pattern, script_constant in scripts_to_check.values():
        if re.search(pattern, text):
            return transliterate(text, script_constant, sanscript.ITRANS)
    return text

def sanitize_lyrics(text: str) -> str | None:
    if not text:
        return None
    lines = text.split('\n')
    if len(lines) > 1 and "Lyrics" in lines[0]:
        lines.pop(0)
    text = '\n'.join(lines)
    text = re.sub(r'\d+ Contributors?Lyrics', '', text)
    text = re.sub(r'Embed$', '', text.strip(), flags=re.IGNORECASE)
    text = html.unescape(text)
    text = unidecode(text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else None

def search_jiosaavn_link(song_name):
    query = f"{song_name} lyrics site:jiosaavn.com"
    try:
        results = list(search(query, num_results=5, sleep_interval=1))
        for link in results:
            if "jiosaavn.com/lyrics/" in link:
                return link
    except Exception as e:
        print(f"Google search for Saavn failed: {e}")
    return None

def get_lyrics_saavn(url):
    try:
        resp = safe_get(url)
        if not resp:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        lyrics_div = soup.find("div", class_="u-disable-select")
        if not lyrics_div:
            return None
        for br in lyrics_div.find_all("br"):
            br.replace_with("\n")
        return lyrics_div.get_text(separator="\n").strip()
    except Exception as e:
        print(f"Failed to scrape Saavn URL {url}: {e}")
    return None

def get_song_lyrics_saavn(song_name):
    link = search_jiosaavn_link(song_name)
    if not link:
        return None
    return get_lyrics_saavn(link)

def find_lyrics(artist: str, title: str) -> str | None:
    lyrics = None
    if genius_client:
        print(f"🔎 Searching Genius for '{title}' by {artist}...")
        try:
            song = genius_client.search_song(title, artist)
            if song:
                print("✅ Found on Genius.")
                lyrics = song.lyrics
        except Exception as e:
            print(f"⚠️ Genius search failed: {e}")
    if not lyrics:
        print(f"🔎 Not found on Genius. Trying JioSaavn for '{title}'...")
        lyrics = get_song_lyrics_saavn(f"{title} {artist}")
        if lyrics:
            print("✅ Found on JioSaavn.")
    if lyrics:
        lyrics = transliterate_if_needed(lyrics)
        lyrics = sanitize_lyrics(lyrics)
        return lyrics
    print(f"😭 Could not find lyrics for '{title}'.")
    return None

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# =================================================================================
# --- 4. API ---
# =================================================================================

class SongRequest(BaseModel):
    artist: str
    title: str

class MoodResponse(BaseModel):
    mood: str
    probabilities: dict
    lyrics: str

@app.post("/predict_mood", response_model=MoodResponse)
def predict_mood_endpoint(request: SongRequest):
    if not model or not vectorizer:
        raise HTTPException(
            status_code=503,
            detail="ML model or vectorizer not loaded. Try again later."
        )
    lyrics = find_lyrics(request.artist, request.title)
    if not lyrics:
        raise HTTPException(status_code=404, detail="Lyrics not found for this song.")
    processed = preprocess_text(lyrics)
    tfidf = vectorizer.transform([processed])
    predicted_label = model.predict(tfidf)[0]
    probs = model.predict_proba(tfidf)[0]
    probs_dict = {mood: prob for mood, prob in zip(model.classes_, probs)}
    return {
        "mood": predicted_label,
        "probabilities": probs_dict,
        "lyrics": lyrics,
    }

@app.get("/")
def read_root():
    return {"status": "SpotifyMood API v2 is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
