import requests
import re
import html
import csv

import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from config import (
    METADATA_DIR,
    SEARCH_TERMS,
    SEARCH_LIMIT_PER_TERM,
    REQUEST_DELAY_SECONDS,
    MAX_RETRIES,
    BACKOFF_BASE_SECONDS
)


API_URL = "https://commons.wikimedia.org/w/api.php"

HEADERS = {
    "User-Agent": "CNNPictureImporter/0.1 (educational project)"
}

CSV_FIELDS = [
    "class_name",
    "search_term",
    "pageid",
    "title",
    "page_url",
    "image_url",
    "download_url",
    "artist",
    "license",
    "license_url",
    "description",
    "selected"
]


def clean_metadata(value):
    if not value:
        return None

    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def is_nd_license(license_name):
    if not license_name:
        return False

    license_name = license_name.upper()

    return bool(re.search(r"\bND\b|NODERIVS|NODE RIVATIVES", license_name))


def request_commons(params):

    for attempt in range(MAX_RETRIES + 1):
        time.sleep(REQUEST_DELAY_SECONDS)

        response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")

            if retry_after:
                try:
                    wait_time = float(retry_after)
                except ValueError:
                    try:
                        retry_date = parsedate_to_datetime(retry_after)
                        wait_time = max(
                            0,
                            (retry_date - datetime.now(timezone.utc)).total_seconds()
                        )
                    except (TypeError, ValueError):
                        wait_time = BACKOFF_BASE_SECONDS * (2 ** attempt)
            else:
                wait_time = BACKOFF_BASE_SECONDS * (2 ** attempt)

            if attempt == MAX_RETRIES:
                response.raise_for_status()

            print(f"HTTP 429 – Warte {wait_time:.1f} Sekunden (Versuch {attempt + 1}/{MAX_RETRIES})")

            time.sleep(wait_time)
            continue

        response.raise_for_status()

        return response.json()

    raise RuntimeError("Wikimedia API: Maximale Wiederholungszahl erreicht.")


def search_commons(search_term, class_name, limit=500):

    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": search_term,
        "gsrnamespace": 6,
        "gsrlimit": 50,
        "prop": "imageinfo|info",
        "inprop": "url",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 1200,
        "format": "json",
        "formatversion": 2
    }

    results = []
    continuation = {}

    while len(results) < limit:

        request_params = params.copy()
        request_params.update(continuation)

        data = request_commons(request_params)

        pages = data.get("query", {}).get("pages", [])

        for page in pages:
            image_info = page.get("imageinfo", [])

            if not image_info:
                continue

            image = image_info[0]
            metadata = image.get("extmetadata", {})

            license_name = clean_metadata(metadata.get("LicenseShortName", {}).get("value"))

            if is_nd_license(license_name):
                continue

            results.append({
                "class_name": class_name,
                "search_term": search_term,
                "pageid": page.get("pageid"),
                "title": page.get("title"),
                "page_url": page.get("fullurl"),
                "image_url": image.get("url"),
                "download_url": image.get("thumburl") or image.get("url"),
                "artist": clean_metadata(metadata.get("Artist", {}).get("value")),
                "license": license_name,
                "license_url": metadata.get("LicenseUrl", {}).get("value"),
                "description": clean_metadata(metadata.get("ImageDescription", {}).get("value"))
            })

            if len(results) >= limit:
                break

        continuation = data.get("continue", {})

        if not continuation:
            break

    return results


def collect_search_results():
    all_results = []

    for class_name, search_terms in SEARCH_TERMS.items():
        class_results = []
        seen_page_ids = set()
        seen_descriptions = set()

        print(f"\n--- Klasse: {class_name} ---")

        for search_term in search_terms:
            print(f"\nSuche: {search_term}")

            try:
                results = search_commons(search_term=search_term, class_name=class_name, limit=SEARCH_LIMIT_PER_TERM)

            except requests.RequestException as error:
                print(f"Fehler bei '{search_term}': {error}")
                continue

            new_results = 0
            duplicates = 0

            for result in results:
                page_id = result["pageid"]
                description = result["description"]

                if (page_id in seen_page_ids) or (description in seen_descriptions):
                    duplicates += 1
                    continue

                seen_page_ids.add(page_id)
                seen_descriptions.add(description)
                class_results.append(result)
                new_results += 1

            print(f"Gefundene Treffer: {len(results)}")
            print(f"Neue Bilder: {new_results}")
            print(f"Duplikate: {duplicates}")

        print(f"\nErgebnis für {class_name}: {len(class_results)} Bilder")

        all_results.extend(class_results)

    return all_results


def export_search_results(results):
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = METADATA_DIR / "search_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, delimiter=";")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nCSV gespeichert: {csv_path}")


if __name__ == "__main__":
    results = collect_search_results()
    export_search_results(results)
    print(f"\nGesamtergebnisse: {len(results)}")