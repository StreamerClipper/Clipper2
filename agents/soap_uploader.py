"""
agents/soap_uploader.py — Soap Opera Shorts: YouTube Uploader

Called by discord_bot.py when the owner reacts ✅ to a soap clip.
Uploads the approved clip to YouTube Shorts using the existing YouTube
credentials already configured in settings.py.

Can also be run standalone:
    python -m agents.soap_uploader --record '{"clip_path": "...", "job": {...}, "hotspot": {...}}'
"""
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from config.settings import settings

log = logging.getLogger("soap_uploader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _strip_episode(title: str) -> str:
    import re
    # Remove channel suffix like @showtv
    title = re.sub(r'\s*@\w+\s*$', '', title).strip()
    # Remove known episode separators
    for sep in [". Bölüm", " Bölüm", " - ", " | ", " Episode"]:
        if sep in title:
            title = title.split(sep)[0].strip()
            break
    # Remove trailing episode number e.g. "Kızılcık Şerbeti 129"
    title = re.sub(r'\s+\d+$', '', title).strip()
    return title


def build_title(job: dict, hotspot: dict, clip_index: int = 1) -> str:
    raw  = job.get("title", "Dizi")
    show = _strip_episode(raw)
    if len(show) > 50:
        show = show[:47] + "..."
    episode_num = None
    m = re.search(r'(\d+)[.\s]*(?:B\u00f6l\u00fcm|Episode|Ep\.?)', raw, re.IGNORECASE)
    if not m:
        m = re.search(r'(?:B\u00f6l\u00fcm|Episode|Ep\.?)[.\s]*(\d+)', raw, re.IGNORECASE)
    if m:
        episode_num = m.group(1)
    bolum = f"B\u00f6l\u00fcm {episode_num}" if episode_num else "B\u00f6l\u00fcm"
    return f"{show} Highlights {bolum} #{clip_index} #Shorts"


def build_description(job: dict, hotspot: dict) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    show    = _strip_episode(job.get("title", "Dizi"))
    tags = build_tags(job)
    # Format each tag as a proper hashtag
    hashtag_str = " ".join(
        f"#{t.lstrip('#').replace(' ', '')}" for t in tags
    )

    if not api_key:
        return f"{show} — en çok tekrar izlenen an.\n\n{hashtag_str}"

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{"role": "user", "content": (
                f"Türk dizisi '{show}' için YouTube Shorts açıklaması yaz.\n"
                "Max 120 karakter, Türkçe, spoiler verme, merak uyandır.\n"
                "Sadece açıklama metnini yaz, başka hiçbir şey yazma."
            )}],
        )
        body = message.content[0].text.strip()
        return f"{body}\n\n{hashtag_str}"
    except Exception as e:
        log.warning(f"Claude description failed: {e}")
        return f"{show} — en çok tekrar izlenen an.\n\n{hashtag_str}"


def build_tags(job: dict) -> list[str]:
    show  = _strip_episode(job.get("title", "Dizi"))
    slug  = show.replace(" ", "")
    words = [w for w in show.split() if len(w) > 2][:4]
    return [show, slug] + words + ["shorts", "dizi", "t\u00fcrkdizisi", "t\u00fcrkdizileri"]


def download_from_discord(message_id: str, dest: Path) -> bool:
    token = settings.DISCORD_BOT_TOKEN
    if not token:
        log.error("DISCORD_BOT_TOKEN not set")
        return False
    SOAP_CLIPS_CHANNEL_ID = "1484834736257106020"
    try:
        import requests
        resp = requests.get(
            f"https://discord.com/api/v10/channels/{SOAP_CLIPS_CHANNEL_ID}/messages/{message_id}",
            headers={"Authorization": f"Bot {token}"},
            timeout=10,
        )
        if resp.status_code != 200:
            log.error(f"Could not fetch Discord message {message_id}: {resp.status_code}")
            return False
        attachments = resp.json().get("attachments", [])
        if not attachments:
            log.error(f"No attachments on Discord message {message_id}")
            return False
        video_url = attachments[0]["url"]
        log.info(f"Downloading clip from Discord: {video_url[:80]}...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(video_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
        size_mb = dest.stat().st_size / 1024 / 1024
        log.info(f"Downloaded {size_mb:.1f}MB \u2192 {dest.name}")
        return True
    except Exception as e:
        log.error(f"Discord download failed: {e}")
        return False


def upload_to_youtube(clip_path: Path, title: str, description: str, tags: list[str]) -> str | None:
    import os
    # Use dizikliper-specific credentials if available, otherwise fall back to default
    client_id     = os.getenv("SOAP_YOUTUBE_CLIENT_ID")     or settings.YOUTUBE_CLIENT_ID
    client_secret = os.getenv("SOAP_YOUTUBE_CLIENT_SECRET") or settings.YOUTUBE_CLIENT_SECRET
    refresh_token = os.getenv("SOAP_YOUTUBE_REFRESH_TOKEN") or settings.YOUTUBE_REFRESH_TOKEN

    if os.getenv("SOAP_YOUTUBE_REFRESH_TOKEN"):
        log.info("Using dizikliper channel credentials")
    else:
        log.warning("SOAP_YOUTUBE_REFRESH_TOKEN not set — using default channel")

    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        log.error("google-api-python-client not installed")
        return None

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=["https://www.googleapis.com/auth/youtube.upload"],
    )
    creds.refresh(Request())
    youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
    body = {
        "snippet": {"title": title, "description": description, "tags": tags, "categoryId": "24", "defaultLanguage": "tr"},
        "status": {"privacyStatus": os.getenv("YT_UPLOAD_PRIVACY", "public"), "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(str(clip_path), mimetype="video/mp4", resumable=True, chunksize=10*1024*1024)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            log.info(f"Upload {int(status.progress()*100)}%...")
    video_id = response["id"]
    log.info(f"Uploaded: https://youtube.com/shorts/{video_id}")
    return video_id


def handle_approval(record: dict) -> str | None:
    job       = record["job"]
    hotspot   = record["hotspot"]
    raw_index = record.get("clip_index", 0)
    clip_number = raw_index if raw_index >= 1 else raw_index + 1
    tmp_path = Path(f"/tmp/soap_clip_{record['message_id']}.mp4")
    if not tmp_path.exists():
        log.info("Clip not on disk \u2014 downloading from Discord attachment...")
        if not download_from_discord(str(record["message_id"]), tmp_path):
            return None
    title       = build_title(job, hotspot, clip_index=clip_number)
    description = build_description(job, hotspot)
    tags        = build_tags(job)
    log.info(f"Uploading: {title}")
    try:
        video_id = upload_to_youtube(tmp_path, title, description, tags)
    finally:
        tmp_path.unlink(missing_ok=True)
    if video_id:
        return f"https://youtube.com/shorts/{video_id}"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    args = parser.parse_args()
    record = json.loads(args.record)
    url = handle_approval(record)
    if url:
        log.info(f"Uploaded: {url}")
        sys.exit(0)
    else:
        log.error("Upload failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
