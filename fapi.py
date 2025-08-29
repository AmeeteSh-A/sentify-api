import os
import re
import html
import joblib
import uvicorn
import nltk
import cloudscraper

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
# Load the Genius API token from an environment variable for security
GENIUS_API_TOKEN = os.getenv("GENIUS_API_TOKEN", "")

# Global objects
model = None
vectorizer = None
genius_client = None

# ==============================
# Startup event
# ==============================
@app.on_event("startup")
def startup_event():
    """
    Initializes necessary components when the API starts.
    NLTK data is now pre-downloaded by the build script.
    """
    global model, vectorizer, genius_client

    print("🔹 Starting SpotifyMood API...")

    # The NLTK download logic has been removed from this function.
    # It is now handled by the build.sh script to prevent race
    # conditions and crashes during startup on Render.

    # Load ML components
    try:
        model = joblib.load("mood_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        print("✅ ML model & vectorizer loaded")
    except Exception as e:
        print(f"⚠️ Failed to load ML model/vectorizer: {e}")
        model = None
        vectorizer = None

    # Initialize Genius client for lyrics fetching
    if GENIUS_API_TOKEN:
        try:
            # Use cloudscraper to bypass Cloudflare protection on Genius.com
            scraper = cloudscraper.create_scraper()
            genius_client = lyricsgenius.Genius(
                GENIUS_API_TOKEN,
                verbose=False,
                remove_section_headers=True,
                timeout=15,
                session=scraper  # Pass the scraper as the request session
            )
            print("✅ Genius client initialized with cloudscraper")
        except Exception as e:
            print(f"⚠️ Genius client initialization failed: {e}")
            genius_client = None
    else:
        print("⚠️ Genius API token not set. Lyrics fetching will be disabled.")

# ==============================
# Helper Functions
# ==============================
def transliterate_if_needed(text: str) -> str:
    """
    Transliterates text from Indic scripts (like Devanagari) to a Latin script format.
    """
    # Define regex patterns for different Indic scripts
    scripts_to_check = {
        "Devanagari": (r'[\u0900-\u097F]', sanscript.DEVANAGARI),
        "Gurmukhi":   (r'[\u0A00-\u0A7F]', sanscript.GURMUKHI),
    }
    for script_name, (pattern, script_const) in scripts_to_check.items():
        # Corrected the line below by removing the extra dot
        if re.search(pattern, text):
            print(f"Detected {script_name} script, transliterating...")
            return transliterate(text, script_const, sanscript.ITRANS)
    return text

def sanitize_lyrics(text: str) -> str | None:
    """
    Cleans up raw lyric text by removing metadata, extra spaces, and decoding HTML entities.
    """
    if not text:
        return None
    # Remove annotations like [Chorus], [Verse], etc. and decode HTML entities
    text = re.sub(r'\[.*?\]', '', unidecode(html.unescape(text)))
    # Replace multiple whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else None

def preprocess_text(text: str) -> str:
    """
    Prepares text for the machine learning model by normalizing, removing stopwords, and lemmatizing.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # Lemmatize and remove stopwords
    processed_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(processed_tokens)

# ==============================
# Lyrics Search Function
# ==============================
def find_lyrics(artist: str, title: str) -> str | None:
    """
    Searches for song lyrics exclusively using the Genius API.
    """
    if not genius_client:
        print("⚠️ Genius client is not available. Cannot fetch lyrics.")
        return None

    try:
        print(f"Searching Genius for '{title}' by {artist}...")
        song = genius_client.search_song(title, artist)
        if song and song.lyrics:
            print("✅ Lyrics found on Genius.")
            lyrics = transliterate_if_needed(song.lyrics)
            return sanitize_lyrics(lyrics)
        else:
            print("❌ Lyrics not found on Genius.")
            return None
    except Exception as e:
        # Added detailed error logging to see the exact problem
        print(f"⚠️ An error occurred during Genius search: {type(e).__name__}: {e}")
        return None

# ==============================
# API Pydantic Models
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
    """
    Receives artist and title, finds lyrics, and predicts the mood of the song.
    """
    if not model or not vectorizer:
        raise HTTPException(status_code=503, detail="ML model or vectorizer is not loaded.")

    lyrics = find_lyrics(request.artist, request.title)
    if not lyrics:
        raise HTTPException(status_code=404, detail="Lyrics could not be found for the specified song.")

    processed_lyrics = preprocess_text(lyrics)
    tfidf_features = vectorizer.transform([processed_lyrics])

    predicted_mood = model.predict(tfidf_features)[0]
    probabilities = model.predict_proba(tfidf_features)[0]
    
    # Create a dictionary of mood probabilities
    prob_dict = dict(zip(model.classes_, probabilities))

    return {
        "mood": predicted_mood,
        "probabilities": prob_dict,
        "lyrics": lyrics
    }

@app.get("/")
def read_root():
    """
    Root endpoint to check if the API is running.
    """
    return {"status": "SpotifyMood API v2 is running"}

# ==============================
# Main Execution Block
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
