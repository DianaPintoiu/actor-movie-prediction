import requests
import pandas as pd
import time
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9"
}


def get_soup(url):
    """Safe HTML fetch with timeout & headers."""
    try:
        time.sleep(1.2)
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except:
        return None


def find_wikipedia_page(title, max_retries=3):
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": title + " film",
        "format": "json"
    }

    for attempt in range(max_retries):
        try:
            r = requests.get(api_url, params=params, headers=HEADERS, timeout=10)

            if "text/html" in r.headers.get("Content-Type", ""):
                print(f"[WARN] Rate limit for '{title}'. Retry {attempt+1}/{max_retries}...")
                time.sleep(3 + attempt * 2)
                continue

            try:
                data = r.json()
            except:
                print(f"[WARN] JSON decode failed for '{title}'. Retry {attempt+1}/{max_retries}...")
                time.sleep(2)
                continue

            if "query" not in data or "search" not in data["query"]:
                return None

            search_results = data["query"]["search"]
            if len(search_results) == 0:
                return None

            page_title = search_results[0]["title"]
            return "https://en.wikipedia.org/wiki/" + page_title.replace(" ", "_")

        except Exception as e:
            print(f"[ERROR] Wikipedia error for '{title}' (attempt {attempt+1}): {e}")
            time.sleep(2)

    print(f"[FAIL] Could not get Wikipedia page for '{title}' after {max_retries} retries.")
    return None


def extract_wikipedia_info(url):
    """
    Extract Budget, Language, Box Office, Country from Wikipedia infobox.
    """
    soup = get_soup(url)
    if soup is None:
        return None, None, None, None

    infobox = soup.select_one("table.infobox")
    if not infobox:
        return None, None, None, None

    budget = language = box_office = country = None

    for row in infobox.select("tr"):
        th = row.find("th")
        td = row.find("td")

        if not th or not td:
            continue

        header = th.text.strip().lower()
        value = td.text.strip()

        if "budget" in header:
            budget = value
        elif "language" in header:
            language = value
        elif "box office" in header:
            box_office = value
        elif "country" in header:
            country = value

    return budget, language, box_office, country


def scrape_wikipedia_for_all(imdb_csv, output_csv="wikipedia_all.csv"):
    df = pd.read_csv(imdb_csv)
    results = []

    for index, title in enumerate(df["title"]):

        wiki_url = find_wikipedia_page(title)

        if not wiki_url:
            results.append([title, None, None, None, None])
            continue


        budget, language, box_office, country = extract_wikipedia_info(wiki_url)

        results.append([title, budget, language, box_office, country])

        print(f"  → Budget: {budget}")
        print(f"  → Language: {language}")
        print(f"  → Box Office: {box_office}")
        print(f"  → Country: {country}")

    out = pd.DataFrame(results, columns=["title", "budget", "language", "box_office", "country"])
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"\n Saved in : {output_csv}")


def main():
    scrape_wikipedia_for_all("movies.csv")
