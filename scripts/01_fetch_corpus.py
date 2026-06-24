"""
scripts/01_fetch_corpus.py

Downloads a topical set of Wikipedia articles to build the CITADEL corpus.
Free, no API key required. Produces ~200+ pages of clean text.

Default topic: "Space exploration" cluster of ~45-55 articles.
Swap TOPIC_SEEDS below to change domain (e.g. cybersecurity, climate science).
"""

import json
import os
import time

import wikipedia
from tqdm import tqdm

wikipedia.set_lang("en")

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Seed pages — we pull these PLUS their linked pages (filtered) to hit 200+ pages.
TOPIC_SEEDS = [
    "Space exploration",
    "NASA",
    "International Space Station",
    "SpaceX",
    "Apollo program",
    "Mars rover",
    "Hubble Space Telescope",
    "James Webb Space Telescope",
    "Voyager program",
    "International Space Station program",
    "Artemis program",
    "Space Shuttle",
    "Low Earth orbit",
    "Geostationary orbit",
    "Rocket",
    "Falcon 9",
    "Saturn V",
    "Apollo 11",
    "Apollo 13",
    "Space race",
    "Sputnik 1",
    "Yuri Gagarin",
    "Neil Armstrong",
    "Cassini–Huygens",
    "New Horizons",
    "Curiosity (rover)",
    "Perseverance (rover)",
    "Mars Exploration Rover",
    "Spacecraft propulsion",
    "Ion thruster",
    "Reusable launch system",
    "Commercial spaceflight",
    "Blue Origin",
    "Space station",
    "Skylab",
    "Mir (space station)",
    "Orbital mechanics",
    "Spaceflight",
    "Astronaut",
    "European Space Agency",
    "Roscosmos",
    "China National Space Administration",
    "Chang'e program",
    "ISRO",
    "Chandrayaan-3",
    "Moon landing",
    "Space probe",
    "Satellite",
    "Geocentric orbit",
    "Spacecraft",
    "Solar System exploration",
]


def fetch_page(title: str):
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return {
            "title": page.title,
            "url": page.url,
            "content": page.content,
        }
    except wikipedia.exceptions.DisambiguationError as e:
        # Take the first disambiguation option as a fallback
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            return {"title": page.title, "url": page.url, "content": page.content}
        except Exception:
            return None
    except Exception as e:
        print(f"  [skip] {title}: {e}")
        return None


def main():
    fetched = []
    seen_titles = set()

    print(f"Fetching {len(TOPIC_SEEDS)} seed articles...")
    for title in tqdm(TOPIC_SEEDS):
        if title in seen_titles:
            continue
        result = fetch_page(title)
        if result and result["title"] not in seen_titles:
            fetched.append(result)
            seen_titles.add(result["title"])
        time.sleep(0.2)  # be polite to the API

    total_words = sum(len(d["content"].split()) for d in fetched)
    # Roughly 500 words per "page" for the 200+ page target
    approx_pages = total_words / 500

    print(f"\nFetched {len(fetched)} articles, ~{total_words:,} words, "
          f"~{approx_pages:.0f} pages worth of text.")

    out_path = os.path.join(OUTPUT_DIR, "corpus.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in fetched:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Saved corpus to {out_path}")


if __name__ == "__main__":
    main()
