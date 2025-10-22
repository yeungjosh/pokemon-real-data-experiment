"""
FAST scraper using Pokemon Showdown's search API.

Instead of trying sequential IDs, use the search endpoint which returns
batches of 51 recent replays at a time.
"""

import requests
import json
import time
from pathlib import Path
from datetime import datetime
from scrape_replays import extract_battle_data, fetch_replay

# Configuration
TIER = "gen9ou"
TARGET_REPLAYS = 5000  # Scale up for reliable training
OUTPUT_DIR = Path("data/replays")
RATE_LIMIT_DELAY = 0.5  # Can be faster with search API

def fetch_recent_battle_ids(tier: str, page: int = 1) -> list[str]:
    """Fetch recent battle IDs using search API.

    Returns up to 51 battle IDs per page.
    """
    url = f"https://replay.pokemonshowdown.com/search.json?format={tier}&page={page}"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            battles = response.json()
            return [b['id'] for b in battles]
        return []
    except Exception as e:
        print(f"Error fetching search page {page}: {e}")
        return []


def scrape_with_search_api(target_count: int):
    """Scrape replays using search API (MUCH faster)."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "battles_fast.jsonl"

    scraped_count = 0
    attempt_count = 0
    page = 1

    print(f"Fast scraping with search API")
    print(f"Target: {target_count} replays")
    print("-" * 60)

    with open(output_file, 'w') as f:
        while scraped_count < target_count:
            # Fetch batch of battle IDs
            print(f"Fetching page {page}...")
            battle_ids = fetch_recent_battle_ids(TIER, page)

            if not battle_ids:
                print(f"No more battles found at page {page}")
                break

            print(f"  Got {len(battle_ids)} battle IDs, processing...")

            # Process each battle
            for battle_id in battle_ids:
                if scraped_count >= target_count:
                    break

                replay = fetch_replay(battle_id)
                attempt_count += 1

                if replay:
                    battle_data = extract_battle_data(replay)
                    if battle_data:
                        f.write(json.dumps(battle_data) + '\n')
                        f.flush()
                        scraped_count += 1

                        if scraped_count % 10 == 0:
                            success_rate = (scraped_count / attempt_count) * 100
                            print(f"  ✓ {scraped_count}/{target_count} replays | "
                                  f"Success rate: {success_rate:.1f}%")

                time.sleep(RATE_LIMIT_DELAY)

            page += 1

    print("-" * 60)
    print(f"✓ Scraping complete!")
    print(f"  Valid replays: {scraped_count}")
    print(f"  Total attempts: {attempt_count}")
    print(f"  Success rate: {(scraped_count/attempt_count)*100:.1f}%")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    scrape_with_search_api(TARGET_REPLAYS)
