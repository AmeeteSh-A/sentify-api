import requests
from bs4 import BeautifulSoup
import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_lyrics_and_mood(song_name: str):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

        search_url = f"https://genius.com/api/search/multi?per_page=5&q={song_name}"
        response = requests.get(search_url, headers=headers, timeout=10)

        if response.status_code != 200:
            logger.error(f"Search request failed: {response.status_code}")
            return None, None

        data = response.json()
        hits = data.get("response", {}).get("sections", [])[0].get("hits", [])
        if not hits:
            logger.warning("No results found on Genius")
            return None, None

        song_url = hits[0]["result"]["url"]
        logger.info(f"Found song URL: {song_url}")

        # Fetch lyrics page
        lyrics_page = requests.get(song_url, headers=headers, timeout=10)

        if lyrics_page.status_code != 200:
            logger.error(f"Lyrics page failed: {lyrics_page.status_code}")
            return None, None

        # --- Debugging Captcha or Bot Blocks ---
        if "captcha" in lyrics_page.text.lower() or "verify you are human" in lyrics_page.text.lower():
            logger.error("Blocked by Genius (Captcha detected)")
            logger.debug(f"Response body:\n{lyrics_page.text[:1000]}")  # log first 1000 chars
            return None, "Blocked by Genius (Captcha)"

        soup = BeautifulSoup(lyrics_page.text, "html.parser")
        lyrics_divs = soup.find_all("div", {"data-lyrics-container": "true"})
        lyrics = "\n".join([div.get_text(separator="\n") for div in lyrics_divs])

        if not lyrics.strip():
            logger.warning("Lyrics not found in parsed HTML")
            return None, None

        # --- Simple Mood Detection (placeholder) ---
        mood = "happy" if re.search(r"\bhappy\b|\blove\b|\bsmile\b", lyrics, re.I) else "sad"

        return lyrics, mood

    except Exception as e:
        logger.exception(f"Error fetching lyrics: {e}")
        return None, None
