import csv
import io
import re
import time
import requests
import cairosvg

from pathlib import Path
from PIL import Image, ImageOps

from config import (
    METADATA_DIR,
    REVIEW_DIR,
    REQUEST_DELAY_SECONDS,
    MAX_RETRIES,
    BACKOFF_BASE_SECONDS
)

CSV_INPUT = METADATA_DIR / "search_results.csv"
CSV_OUTPUT = METADATA_DIR / "download_results.csv"

HEADERS = {
    "User-Agent": "CNNPictureImporter/0.1 (educational project) - Contact: marcel-kramer@gmx.de"
}


def download_image(url):

    for attempt in range(MAX_RETRIES + 1):

        time.sleep(REQUEST_DELAY_SECONDS)

        try:
            response = requests.get(url, headers=HEADERS, timeout=60)

            if response.status_code == 429:

                retry_after = response.headers.get("Retry-After")

                try:
                    wait_time = float(retry_after)
                except (TypeError, ValueError):
                    wait_time = BACKOFF_BASE_SECONDS * (2 ** attempt)

                if attempt == MAX_RETRIES:
                    response.raise_for_status()

                print(f"HTTP 429 - Warte {wait_time:.1f} Sekunden")
                time.sleep(wait_time)
                continue

            response.raise_for_status()

            return response.content

        except requests.RequestException:

            if attempt == MAX_RETRIES:
                raise

            wait_time = BACKOFF_BASE_SECONDS * (2 ** attempt)

            print(f"Download fehlgeschlagen. Neuer Versuch in {wait_time}s.")
            time.sleep(wait_time)

    raise RuntimeError("Download nach mehreren Versuchen fehlgeschlagen.")


def save_image(image_data, output_path):

    if (image_data.lstrip().startswith(b"<?xml")) or (image_data.lstrip().startswith(b"<svg")):
        image_data = cairosvg.svg2png(bytestring=image_data)

    with Image.open(io.BytesIO(image_data)) as image:
        image = ImageOps.exif_transpose(image)

        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            image = image.convert("RGBA")

            background = Image.new("RGBA", image.size, (255, 255, 255, 255))

            background.alpha_composite(image)
            image = background.convert("RGB")

        else:
            image = image.convert("RGB")

        image.save(output_path, format="JPEG", quality=90, optimize=True)


def export_download_result(result):

    file_exists = CSV_OUTPUT.exists()

    with open(CSV_OUTPUT, "a", newline="", encoding="utf-8-sig") as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=result.keys(), delimiter=";")

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)


def normalize_search_term (term: str):
    return term.replace(" ", "_")


def download_all_images():
    with open(CSV_INPUT, "r", newline="", encoding="utf-8-sig") as csv_file:

        reader = csv.DictReader(csv_file, delimiter=";")

        for index, row in enumerate(reader, start=1):
            class_name = row["class_name"]
            search_term = row["search_term"]
            page_id = row["pageid"]

            if row["title"].split(".")[-1] in ["pdf", "ogg"]:
                continue

            search_folder = normalize_search_term(search_term)

            output_dir = REVIEW_DIR / class_name / search_folder
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = output_dir / f"{page_id}.jpg"

            print(f"\n[{index}] {row['title']}")

            if output_path.exists():
                continue

            try:
                url = row.get("download_url") or row.get("image_url")

                image_data = download_image(url)

                save_image(image_data, output_path)

                row["download_status"] = "SUCCESS"
                row["local_file"] = str(output_path)
                row["download_error"] = None

                print("Success")

            except Exception as error:

                row["download_status"] = "ERROR"
                row["local_file"] = None
                row["download_error"] = str(error)

                print(f"Fehler: {error}")

            export_download_result(row)

    print("\nDownload abgeschlossen.")


if __name__ == "__main__":
    download_all_images()
