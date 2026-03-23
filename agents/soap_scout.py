"""
agents/soap_scout.py — Soap Opera Shorts: Scout

Queues YouTube episode URLs for processing.
All yt-dlp work (metadata, hotspot detection, downloading) happens in
GitHub Actions via soap_clipper.py where Node.js is available.

Run on PythonAnywhere always-on task (for playlist polling):
    cd /home/StreamerClipper/clipbot && python -m agents.soap_scout

Or triggered as subprocess by discord_bot.py on !soap clip <url>:
    python -m agents.soap_scout --url https://www.youtube.com/watch?v=XXXXX
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from config.settings import settings

log = logging.getLogger("soap_scout")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

POLL_INTERVAL     = int(os.getenv("SOAP_POLL_INTERVAL", 3600))
SOAP_PENDING_FILE = Path("output/soap_pending.jsonl")
SOAP_SEEN_FILE    = Path("output/soap_seen.json")

# Soap-specific Discord channels
SOAP_CLIPS_CHANNEL_ID = "1484834736257106020"
SOAP_LOG_CHANNEL_ID   = "1484834748181385256"
SOAP_INPUT_CHANNEL_ID = "1484842601617293394"


# =============================================================================
# Discord logging
# =============================================================================

def discord_log(message: str, channel_id: str = None):
    token = settings.DISCORD_BOT_TOKEN
    if not token:
        return
    cid = channel_id or SOAP_LOG_CHANNEL_ID
    try:
        requests.post(
            f"https://discord.com/api/v10/channels/{cid}/messages",
            headers={"Authorization": f"Bot {token}"},
            json={"content": message},
            timeout=5,
        )
    except Exception:
        pass


# =============================================================================
# Seen-video tracking (for playlist polling)
# =============================================================================

def load_seen() -> set[str]:
    if SOAP_SEEN_FILE.exists():
        return set(json.loads(SOAP_SEEN_FILE.read_text()))
    return set()


def mark_seen(video_id: str):
    seen = load_seen()
    seen.add(video_id)
    SOAP_SEEN_FILE.write_text(json.dumps(list(seen)))


# =============================================================================
# Queue + git push (triggers GitHub Actions)
# =============================================================================

def process_url(url: str, mute: bool = False) -> bool:
    log.info(f"Queuing: {url}")

    # Fetch title for commit message
    title = url  # fallback
    try:
        result = subprocess.run(
            ["yt-dlp", "--get-title", "--no-warnings", "--cookies", "cookies.txt", url],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            title = result.stdout.strip()
    except Exception:
        pass

    SOAP_PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)
    job = {
        "url":       url,
        "title":     title,
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "mute":      mute,
    }
    with open(SOAP_PENDING_FILE, "a") as f:
        f.write(json.dumps(job) + "\n")

    # Trigger GitHub Actions via a tracked trigger file
    # (soap_pending.jsonl is gitignored so we can't commit it directly)
    cwd = Path("/home/StreamerClipper/clipbot")
    try:
        # Stash FIRST before touching anything
        subprocess.run(["git", "stash"], cwd=cwd, check=False)

        # Write trigger file
        trigger = cwd / ".soap_trigger"
        trigger.write_text(json.dumps({
            "url":       url,
            "queued_at": job["queued_at"],
            "mute":      mute,
            "ts":        datetime.now(timezone.utc).isoformat(),
        }) + "\n")

        subprocess.run(["git", "add", ".soap_trigger"], cwd=cwd, check=True)
        subprocess.run(["git", "commit", "-m", f"[soap] queue: {url}"], cwd=cwd, check=True)
        subprocess.run(["git", "fetch", "origin", "main"], cwd=cwd, check=True)
        subprocess.run(["git", "rebase", "origin/main"], cwd=cwd, check=True)
        subprocess.run(["git", "stash", "pop"], cwd=cwd, check=False)
        # Retry push up to 3 times in case Actions pushed between our rebase and push
        for attempt in range(3):
            result = subprocess.run(["git", "push", "origin", "main"], cwd=cwd, capture_output=True)
            if result.returncode == 0:
                break
            log.warning(f"Push attempt {attempt+1} failed — pulling and retrying...")
            subprocess.run(["git", "pull", "--rebase", "origin", "main"], cwd=cwd, capture_output=True)
        else:
            raise subprocess.CalledProcessError(1, "git push")
        log.info("Pushed to GitHub — Actions triggered")
        discord_log(
            f"📋 **[Soap]** Episode queued — fetching hotspots and generating clips in GitHub Actions...\n`{url}`",
            channel_id=SOAP_LOG_CHANNEL_ID,
        )
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"Git push failed: {e}")
        discord_log(f"❌ **[Soap]** Git push failed: `{e}`", channel_id=SOAP_LOG_CHANNEL_ID)
        return False


# =============================================================================
# Playlist polling (optional — for auto-monitoring)
# Uses yt-dlp only for flat playlist fetch (no n-challenge needed)
# =============================================================================

def fetch_playlist_entries(playlist_url: str, max_entries: int = 5) -> list[dict]:
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--flat-playlist",
        "--playlist-end", str(max_entries),
        "--no-warnings",
        "--cookies", "/home/StreamerClipper/clipbot/cookies.txt",
        playlist_url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        log.error(f"yt-dlp playlist fetch failed: {result.stderr[:300]}")
        return []

    entries = []
    for line in result.stdout.strip().splitlines():
        try:
            entry = json.loads(line)
            entries.append({
                "id":  entry.get("id", ""),
                "url": entry.get("url") or f"https://www.youtube.com/watch?v={entry['id']}",
            })
        except Exception:
            continue
    return entries


def poll_playlist(playlist_url: str):
    log.info(f"Polling playlist: {playlist_url} every {POLL_INTERVAL}s")
    discord_log(
        f"📡 **[Soap Scout]** Started — polling playlist every {POLL_INTERVAL//3600}h",
        channel_id=SOAP_LOG_CHANNEL_ID,
    )
    while True:
        seen = load_seen()
        entries = fetch_playlist_entries(playlist_url, max_entries=5)

        for entry in entries:
            vid = entry["id"]
            if vid in seen:
                continue
            log.info(f"New episode found: {vid}")
            ok = process_url(entry["url"])
            if ok:
                mark_seen(vid)

        time.sleep(POLL_INTERVAL)

def queue_character_clip(url: str, character: str):
    """Write .character_trigger and push to GitHub to trigger character_clipper workflow."""
    import json, subprocess
    from datetime import datetime, timezone
    from pathlib import Path

    trigger = Path(".character_trigger")
    data = {
        "url":       url,
        "character": character,
        "title":     url,
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "ts":        datetime.now(timezone.utc).timestamp(),
    }
    trigger.write_text(json.dumps(data))

    try:
        subprocess.run(["git", "stash"], capture_output=True)
        subprocess.run(["git", "pull", "--rebase", "origin", "main"], capture_output=True)
        subprocess.run(["git", "stash", "pop"], capture_output=True)
        subprocess.run(["git", "add", "-f", ".character_trigger"], capture_output=True)
        subprocess.run(["git", "commit", "-m", f"[character] queue: {url} ({character})"], capture_output=True)
        for _ in range(3):
            result = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
            if result.returncode == 0:
                log.info(f"Character job queued: {character} — {url}")
                return
            subprocess.run(["git", "pull", "--rebase", "origin", "main"], capture_output=True)
        log.error("Character git push failed after 3 retries")
    except Exception as e:
        log.error(f"queue_character_clip failed: {e}")

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Soap Opera Shorts — Scout")
    parser.add_argument("--url",      help="Queue a single YouTube URL immediately")
    parser.add_argument("--playlist", help="Poll a playlist URL")
    parser.add_argument("--debug",    action="store_true")
    parser.add_argument("--mute", action="store_true", help="Mute audio to avoid copyright")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.url:
        ok = process_url(args.url, mute=args.mute)
        sys.exit(0 if ok else 1)

    playlist_url = args.playlist or os.getenv("SOAP_PLAYLIST_URL")
    if not playlist_url:
        log.error("No --url or --playlist provided, and SOAP_PLAYLIST_URL not set in .env")
        sys.exit(1)

    poll_playlist(playlist_url)


if __name__ == "__main__":
    main()