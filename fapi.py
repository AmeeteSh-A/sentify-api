import os
import re
import random
import html
import joblib
import uvicorn
import time
import cloudscraper
import nltk

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

# ==============================
# FastAPI app
# ==============================
app = FastAPI(
    title="SpotifyMood API",
    description="API to find lyrics and predict song mood.",
    version="2.0.0"
)

# ==============================
# Configs
# ==============================
GENIUS_API_TOKEN = os.getenv("GENIUS_API_TOKEN", "")
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
]

# Global objects
model = None
vectorizer = None
genius_client = None
scraper = None

# ==============================
# Startup event
# ==============================
@app.on_event("startup")
def startup_event():
    global model, vectorizer, genius_client, scraper

    print("🔹 Starting SpotifyMood API...")

    # Cloudscraper
    try:
        scraper = cloudscraper.create_scraper()
        print("✅ Cloudscraper initialized")
    except Exception as e:
        print(f"⚠️ Cloudscraper failed: {e}")

    # NLTK downloads
    nltk_data_path = "/tmp/nltk_data"
    nltk.data.path.append(nltk_data_path)
    for pkg in ["stopwords", "wordnet"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, download_dir=nltk_data_path)

    # Load ML components
    try:
        model = joblib.load("mood_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        print("✅ ML model & vectorizer loaded")
    except Exception as e:
        print(f"⚠️ Failed to load ML model/vectorizer: {e}")
        model = None
        vectorizer = None

    # Genius client
    if GENIUS_API_TOKEN:
        try:
            genius_client = lyricsgenius.Genius(
                GENIUS_API_TOKEN, verbose=False, remove_section_headers=True
            )
            print("✅ Genius client initialized")
        except Exception as e:
            print(f"⚠️ Genius client failed: {e}")
            genius_client = None
    else:
        print("⚠️ Genius API token not set")

# ==============================
# Helpers
# ==============================
def get_random_headers():
    return {"User-Agent": random.choice(USER_AGENTS)}

def safe_get(url, retries=3):
    for attempt in range(retries):
        try:
            resp = scraper.get(url, headers=get_random_headers(), timeout=10)
            if resp.status_code == 200 and "captcha" not in resp.text.lower():
                return resp
        except Exception as e:
            print(f"Request failed attempt {attempt+1}: {e}")
        time.sleep(2 ** attempt)
    return None

def transliterate_if_needed(text):
    scripts_to_check = {
        "Devanagari": (r'[\u0900-\u097F]', sanscript.DEVANAGARI),
        "Gurmukhi":   (r'[\u0A00-\u0A7F]', sanscript.GURMUKHI),
    }
    for pattern, script_const in scripts_to_check.values():
        if re.search(pattern, text):
            return transliterate(text, script_const, sanscript.ITRANS)
    return text

def sanitize_lyrics(text: str) -> str | None:
    if not text:
        return None
    text = re.sub(r'\[.*?\]', '', unidecode(html.unescape(text)))
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else None

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in tokens if w not in stop_words])

# ==============================
# Lyrics search
# ==============================
def search_jiosaavn_link(song_name):
    query = f"{song_name} lyrics site:jiosaavn.com"
    try:
        for link in search(query, num_results=5, sleep_interval=1):
            if "jiosaavn.com/lyrics/" in link:
                return link
    except Exception as e:
        print(f"Google search failed: {e}")
    return None

def get_lyrics_saavn(url):
    resp = safe_get(url)
    if not resp:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    div = soup.find("div", class_="u-disable-select")
    if not div:
        return None
    for br in div.find_all("br"):
        br.replace_with("\n")
    return div.get_text(separator="\n").strip()

def find_lyrics(artist: str, title: str) -> str | None:
    lyrics = None
    if genius_client:
        try:
            song = genius_client.search_song(title, artist)
            if song:
                lyrics = song.lyrics
        except Exception as e:
            print(f"Genius search failed: {e}")
    if not lyrics:
        link = search_jiosaavn_link(f"{title} {artist}")
        if link:
            lyrics = get_lyrics_saavn(link)
    if lyrics:
        lyrics = transliterate_if_needed(lyrics)
        return sanitize_lyrics(lyrics)
    return None

# ==============================
# API Models
# ==============================
class SongRequest(BaseModel):
    artist: str
    title: str

class MoodResponse(BaseModel):
    mood: str
    probabilities: dict
    lyrics: str

# ==============================
# API Endpoints
# ==============================
@app.post("/predict_mood", response_model=MoodResponse)
def predict_mood_endpoint(request: SongRequest):
    if not model or not vectorizer:
        raise HTTPException(503, "ML model/vectorizer not loaded.")
    lyrics = find_lyrics(request.artist, request.title)
    if not lyrics:
        raise HTTPException(404, "Lyrics not found.")
    processed = preprocess_text(lyrics)
    tfidf = vectorizer.transform([processed])
    pred = model.predict(tfidf)[0]
    probs = model.predict_proba(tfidf)[0]
    return {"mood": pred, "probabilities": dict(zip(model.classes_, probs)), "lyrics": lyrics}

@app.get("/")
def read_root():
    return {"status": "SpotifyMood API v2 is running"}

# ==============================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
