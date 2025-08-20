import os
import re
import random
import html
import joblib
import uvicorn
import requests
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

# =================================================================================
# --- 1. CONFIGURATION & STARTUP ---
# =================================================================================

# --- FastAPI App Initialization ---
app = FastAPI(
    title="SpotifyMood API",
    description="API to find lyrics and predict song mood.",
    version="2.0.0"
)

# --- Genius API Token ---
# IMPORTANT: For security, it's better to load this from an environment variable
# For now, we'll keep it here as in your original script.
GENIUS_API_TOKEN = "lyN7oYYT8zEG_I1eg1ySpesW_1wW9HnYg4Ctv6tZgw-wYoAncBsXYs-sWJZ1l3ES"

# --- Rotating User-Agent Headers ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.90 Safari/537.36",
]

# --- Global objects loaded once at startup ---
model = None
vectorizer = None
genius_client = None

@app.on_event("startup")
def load_models_and_clients():
    """This function runs once when the API starts."""
    global model, vectorizer, genius_client

    # 1. Download necessary NLTK data
    print("Downloading NLTK data...")
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    try:
        WordNetLemmatizer().lemmatize("test")
    except LookupError:
        nltk.download('wordnet')
    
    # 2. Load ML model and vectorizer
    print("Loading ML model and vectorizer...")
    model = joblib.load('mood_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("✅ ML components loaded.")

    # 3. Initialize Genius API client
    if not GENIUS_API_TOKEN or "PASTE" in GENIUS_API_TOKEN:
        print("⚠️ Genius API token is not set. The Genius scraper will be disabled.")
        genius_client = None
    else:
        print("Initializing Genius API client...")
        genius_client = lyricsgenius.Genius(GENIUS_API_TOKEN, verbose=False, remove_section_headers=True)
        print("✅ Genius client initialized.")

# =================================================================================
# --- 2. HELPER FUNCTIONS (Your Scraper & ML Preprocessing) ---
# =================================================================================

def get_random_headers():
    return {"User-Agent": random.choice(USER_AGENTS)}

def transliterate_if_needed(text):
    # (Your transliteration logic here - no changes needed)
    # ...
    scripts_to_check = {
        "Devanagari": (r'[\u0900-\u097F]', sanscript.DEVANAGARI),
        "Gurmukhi":   (r'[\u0A00-\u0A7F]', sanscript.GURMUKHI),
        # Add other scripts as in your original file
    }
    for script_name, (pattern, script_constant) in scripts_to_check.items():
        if re.search(pattern, text):
            return transliterate(text, script_constant, sanscript.ITRANS)
    return text

# --- Enhanced Lyric Sanitization Function ---
def sanitize_lyrics(text: str) -> str | None:
    """
    A more aggressive function to deep clean lyrics from Genius.
    """
    if not text:
        return None

    # 1. Initial Genius-specific cleaning
    # The lyricsgenius library sometimes leaves the song title and "Lyrics" on the first line
    lines = text.split('\n')
    if len(lines) > 1 and "Lyrics" in lines[0]:
        lines.pop(0) # Remove the first line if it looks like a title header
    
    # Re-join the text after potentially removing the first line
    text = '\n'.join(lines)

    # 2. Use regex to remove common leftover Genius junk
    # Removes patterns like "13 Contributors", "2 Contributors", etc.
    text = re.sub(r'\d+ Contributors?Lyrics', '', text)
    # Removes the word "Embed" which is often left at the end
    text = re.sub(r'Embed$', '', text.strip(), flags=re.IGNORECASE)

    # 3. Perform the original sanitization steps
    text = html.unescape(text)
    text = unidecode(text)  # Handles a wide range of characters
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in square brackets (e.g., [Chorus])
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

    # Return None if the cleaning results in an empty string
    return text if text else None
# --- Saavn Scraper ---
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
        response = requests.get(url, headers=get_random_headers(), timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        lyrics_div = soup.find("div", class_="u-disable-select")
        if not lyrics_div: return None
        # Replace <br> tags with newlines for better formatting
        for br in lyrics_div.find_all("br"):
            br.replace_with("\n")
        return lyrics_div.get_text(separator="\n").strip()
    except Exception as e:
        print(f"Failed to scrape Saavn URL {url}: {e}")
    return None

def get_song_lyrics_saavn(song_name):
    link = search_jiosaavn_link(song_name)
    if not link: return None
    return get_lyrics_saavn(link)

# --- Master Lyric Finder Function ---
def find_lyrics(artist: str, title: str) -> str | None:
    """Combines all scraping methods to find lyrics."""
    lyrics = None
    
    # 1. Try Genius API first (if client is available)
    if genius_client:
        print(f"🔎 Searching Genius for '{title}' by {artist}...")
        song = genius_client.search_song(title, artist)
        if song:
            print("✅ Found on Genius.")
            lyrics = song.lyrics

    # 2. Fallback to JioSaavn scraper
    if not lyrics:
        print(f"🔎 Not found on Genius. Trying JioSaavn for '{title}'...")
        lyrics = get_song_lyrics_saavn(f"{title} {artist}")
        if lyrics:
            print("✅ Found on JioSaavn.")

    # 3. Final processing if lyrics were found
    if lyrics:
        transliterated = transliterate_if_needed(lyrics)
        sanitized = sanitize_lyrics(transliterated)
        return sanitized

    print(f"😭 Could not find lyrics for '{title}'.")
    return None

# --- ML Model Text Preprocessing ---
def preprocess_text(text: str) -> str:
    # (Your ML preprocessing logic here - no changes needed)
    # ...
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english')) # Add your custom stopwords here if needed
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# =================================================================================
# --- 3. API DEFINITION ---
# =================================================================================

class SongRequest(BaseModel):
    artist: str
    title: str

class MoodResponse(BaseModel):
    mood: str
    probabilities: dict
    lyrics: str # Also return the found lyrics for debugging/display

@app.post("/predict_mood", response_model=MoodResponse)
def predict_mood_endpoint(request: SongRequest):
    """
    Finds lyrics for a song and predicts its mood.
    """
    # 1. Find the lyrics using the combined scraper
    found_lyrics = find_lyrics(request.artist, request.title)
    
    if not found_lyrics:
        raise HTTPException(status_code=404, detail="Lyrics not found for this song.")

    # 2. Preprocess lyrics for the ML model
    processed_lyrics = preprocess_text(found_lyrics)
    
    # 3. Vectorize and predict
    lyrics_tfidf = vectorizer.transform([processed_lyrics])
    predicted_mood_label = model.predict(lyrics_tfidf)[0]
    mood_probabilities = model.predict_proba(lyrics_tfidf)[0]
    
    # 4. Format the response
    probs_dict = {mood: prob for mood, prob in zip(model.classes_, mood_probabilities)}
    
    return {
        "mood": predicted_mood_label,
        "probabilities": probs_dict,
        "lyrics": found_lyrics, # Return the lyrics we found
    }

@app.get("/")
def read_root():
    return {"status": "SpotifyMood API v2 is running"}

# --- Allows running the script directly with `python fapi.py` ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)