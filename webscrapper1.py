import time
import math
import re
import json
import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


START_URL = "https://www.imdb.com/search/title/?title_type=feature&sort=num_votes,desc&language=en-US&region=US"


def setup_driver():
    opts = Options()
    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--disable-infobars")
    opts.add_argument("--disable-extensions")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--no-sandbox")

    # 🔥 VERY IMPORTANT for IMDb language
    opts.add_argument("--lang=en-US")
    opts.add_argument("Accept-Language=en-US,en;q=0.9")

    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option('useAutomationExtension', False)

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=opts
    )

    # remove webdriver flag
    driver.execute_cdp_cmd(
        "Network.setExtraHTTPHeaders",
        {"headers": {"Accept-Language": "en-US,en;q=0.9"}}
    )

    return driver


def compute_popularity(rating, votes):
    try:
        return round(float(rating) * math.log1p(int(votes)), 2)
    except:
        return 0


def extract_json_ld(soup):
    tag = soup.find("script", type="application/ld+json")
    if not tag:
        return {}

    try:
        data = json.loads(tag.string)
    except:
        return {}

    out = {
        "rating": data.get("aggregateRating", {}).get("ratingValue"),
        "votes": data.get("aggregateRating", {}).get("ratingCount")
    }

    # year
    date = data.get("datePublished")
    if date:
        m = re.search(r"\d{4}", date)
        out["release_year"] = int(m.group(0)) if m else None
    else:
        out["release_year"] = None

    # genres
    genres = data.get("genre")
    if isinstance(genres, list):
        out["genres"] = "|".join(genres)
    else:
        out["genres"] = genres or ""

    return out


def extract_runtime_minutes(driver):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.7)")
    time.sleep(0.8)

    soup = BeautifulSoup(driver.page_source, "html.parser")

    tag = soup.select_one("li[data-testid='title-techspec_runtime'] span")
    if not tag:
        return None

    text = tag.get_text(strip=True).lower()
    h = re.search(r"(\d+)h", text)
    m = re.search(r"(\d+)m", text)
    hours = int(h.group(1)) if h else 0
    mins = int(m.group(1)) if m else 0

    return hours * 60 + mins


def extract_cast_and_director(soup):
    tag = soup.select_one("li[data-testid='title-pc-principal-credit'] a")
    director = tag.get_text(strip=True) if tag else ""

    items = soup.select("div[data-testid='title-cast-item']")
    actors = []
    roles = []

    for i, item in enumerate(items):
        nm = item.select_one("a[data-testid='title-cast-item__actor']")
        if nm:
            actors.append(nm.text.strip())
            roles.append("Principal" if i < 3 else "Supporting")

    return director, actors, roles


# --------------------------
# 🔥 FUNCTION CARE SCRAPE-UIE 3000 FILME CU LOAD MORE
# --------------------------

def scrape_imdb(target_count=3000, output="movies.csv"):
    driver = setup_driver()
    wait = WebDriverWait(driver, 10)

    print("Opening IMDb…")
    driver.get(START_URL)
    time.sleep(3)

    all_links = set()
    batch_index = 1

    while len(all_links) < target_count:

        print(f"\n🔵 Batch {batch_index}: scraping current 50 movies on screen...")

        soup = BeautifulSoup(driver.page_source, "html.parser")

        page_links = [
            "https://www.imdb.com" + a["href"].split("?")[0]
            for a in soup.select("li.ipc-metadata-list-summary-item a[href*='/title/']")
        ]

        print("   Found:", len(page_links), "movies on page")

        for link in page_links:
            all_links.add(link)

        print(f"   Total collected so far: {len(all_links)}")

        # dacă am ajuns la target → STOP
        if len(all_links) >= target_count:
            break

        # CLICK pe "Load more"
        try:
            btn = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button//span[contains(text(),'50 more')]/.."))
            )
            print("   ➤ Clicking LOAD MORE...")
            driver.execute_script("arguments[0].click();", btn)
            time.sleep(2)
        except:
            print("❌ No more load more button!")
            break

        batch_index += 1

    # convertim în listă
    all_links = list(all_links)[:target_count]
    print("\n====================")
    print("Total movie links:", len(all_links))
    print("====================\n")

    # -------------------------
    # Scrape fiecare film
    # -------------------------
    movies = []

    for i, url in enumerate(all_links, 1):
        print(f"[{i}/{len(all_links)}] → {url}")

        driver.get(url)
        time.sleep(1.1)

        soup_movie = BeautifulSoup(driver.page_source, "html.parser")

        jd = extract_json_ld(soup_movie)
        director, actors, roles = extract_cast_and_director(soup_movie)
        runtime = extract_runtime_minutes(driver)

        rating = jd.get("rating")
        votes = jd.get("votes")
        popularity = compute_popularity(rating, votes)

        h1 = soup_movie.select_one("h1")
        title = h1.get_text(strip=True) if h1 else ""

        movies.append({
            "title": title,
            "url": url,
            "rating": rating,
            "votes": votes,
            "release_year": jd.get("release_year"),
            "genres": jd.get("genres"),
            "runtime_minutes": runtime,
            "director": director,
            "actors": "|".join(actors),
            "roles": "|".join(roles),
            "popularity_index": popularity
        })

        pd.DataFrame(movies).to_csv(output, index=False)

    driver.quit()
    return pd.DataFrame(movies)


def main():
    scrape_imdb(target_count=3000)


if __name__ == "__main__":
    main()
