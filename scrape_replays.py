"""
Quick scraper for Pokemon Showdown replays.

Fetches Gen 9 OU replays and extracts teams + battle outcomes.
"""

import requests
import json
import time
from pathlib import Path
from datetime import datetime

# Configuration
TIER = "gen9ou"
MIN_RATING = 1000  # Filter out very low-skill matches (lowered for testing)
TARGET_REPLAYS = 10  # Test run first
RATE_LIMIT_DELAY = 1.0  # seconds between requests
OUTPUT_DIR = Path("data/replays")

# Pokemon Showdown replay API
# Format: https://replay.pokemonshowdown.com/gen9ou-2093847562.json

def fetch_replay(battle_id: str) -> dict | None:
    """Fetch a single replay via JSON API."""
    url = f"https://replay.pokemonshowdown.com/{battle_id}.json"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error fetching {battle_id}: {e}")
        return None


def extract_battle_data(replay: dict) -> dict | None:
    """Extract teams and outcome from replay JSON."""
    try:
        # Parse battle log
        log = replay.get('log', '')
        lines = log.split('\n')

        # Extract player names and ratings from |player| lines
        # Format: |player|p1|username|avatar|rating
        p1_name = None
        p2_name = None
        p1_rating = 1500
        p2_rating = 1500

        p1_team = []
        p2_team = []
        winner = None

        for line in lines:
            parts = line.split('|')

            # Parse player info
            if len(parts) > 1 and parts[0] == '' and parts[1] == 'player':
                if len(parts) < 6:
                    continue
                player_id = parts[2]  # 'p1' or 'p2'
                username = parts[3]
                rating = int(parts[5]) if parts[5].isdigit() else 1500

                if player_id == 'p1':
                    p1_name = username
                    p1_rating = rating
                elif player_id == 'p2':
                    p2_name = username
                    p2_rating = rating

            # Parse team preview
            # Format: |poke|p1|Garchomp, M|
            elif len(parts) > 3 and parts[0] == '' and parts[1] == 'poke':
                player_id = parts[2]
                pokemon_name = parts[3].split(',')[0]  # Remove gender/forme details initially

                if player_id == 'p1':
                    p1_team.append(pokemon_name)
                elif player_id == 'p2':
                    p2_team.append(pokemon_name)

            # Parse winner
            # Format: |win|username
            elif len(parts) > 2 and parts[0] == '' and parts[1] == 'win':
                winner_name = parts[2]
                if winner_name == p1_name:
                    winner = 'p1'
                elif winner_name == p2_name:
                    winner = 'p2'

        # Validation
        if len(p1_team) != 6 or len(p2_team) != 6:
            return None

        if not winner:
            return None  # No clear winner

        # Filter by rating
        if p1_rating < MIN_RATING or p2_rating < MIN_RATING:
            return None

        return {
            'battle_id': replay.get('id', ''),
            'p1_name': p1_name,
            'p2_name': p2_name,
            'p1_team': p1_team,
            'p2_team': p2_team,
            'winner': winner,
            'p1_rating': p1_rating,
            'p2_rating': p2_rating,
            'rating_diff': abs(p1_rating - p2_rating),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        print(f"Error parsing replay: {e}")
        return None


def scrape_replays(start_id: int, target_count: int):
    """Scrape replays starting from a battle ID."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "battles.jsonl"

    scraped_count = 0
    attempt_count = 0
    battle_id = start_id

    print(f"Starting scrape from battle {TIER}-{battle_id}")
    print(f"Target: {target_count} valid replays")
    print(f"Rate limit: {RATE_LIMIT_DELAY}s between requests")
    print("-" * 60)

    with open(output_file, 'a') as f:
        while scraped_count < target_count:
            # Construct battle ID
            full_id = f"{TIER}-{battle_id}"

            # Fetch replay
            replay = fetch_replay(full_id)
            attempt_count += 1

            if replay:
                # Parse and validate
                battle_data = extract_battle_data(replay)

                if battle_data:
                    # Save to JSONL file
                    f.write(json.dumps(battle_data) + '\n')
                    f.flush()  # Ensure data is written

                    scraped_count += 1

                    if scraped_count % 10 == 0:
                        success_rate = (scraped_count / attempt_count) * 100
                        print(f"✓ {scraped_count}/{target_count} replays | "
                              f"Success rate: {success_rate:.1f}% | "
                              f"Battle: {full_id}")

            # Increment battle ID and rate limit
            battle_id += 1
            time.sleep(RATE_LIMIT_DELAY)

    print("-" * 60)
    print(f"✓ Scraping complete!")
    print(f"  Valid replays: {scraped_count}")
    print(f"  Total attempts: {attempt_count}")
    print(f"  Success rate: {(scraped_count/attempt_count)*100:.1f}%")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    # Start from a recent battle ID (found via search API)
    # Format: gen9ou-2466685514 (Oct 2025)
    START_ID = 2466685514

    # Quick test with 10 replays first, then increase TARGET_REPLAYS
    scrape_replays(START_ID, TARGET_REPLAYS)
