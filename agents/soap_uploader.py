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
import sys
from pathlib import Path

from config.settings import settings

log = logging.getLogger("soap_uploader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# =============================================================================
# Build video metadata
# =============================================================================

def ts_label(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def build_title(job: dict, hotspot: dict, clip_index: int = 1) -> str:
    """
    Title format: "<Dizi Adı> Highlights Bölüm <episode> #<clip> #Shorts"
    e.g. "Kızılcık Şerbeti Highlights Bölüm 45 #2 #Shorts"
    clip_index is 1-based (which clip out of 3).
    Episode number is extracted from the job title if present.
    """
    raw   = job.get("title", "Dizi")
    show  = _strip_episode(raw)
    if len(show) > 50:
        show = show[:47] + "..."

    # Try to extract episode number from the original title
    episode_num = None
    import re
    m = re.search(r'(\d+)[.\s]*(?:Bölüm|Episode|Ep\.?)', raw, re.IGNORECASE)
    if not m:
        m = re.search(r'(?:Bölüm|Episode|Ep\.?)[.\s]*(\d+)', raw, re.IGNORECASE)
    if m:
        episode_num = m.group(1)

    bolum = f"Bölüm {episode_num}" if episode_num else "Bölüm"
    return f"{show} Highlights {bolum} #{clip_index} #Shorts"

def build_description(job: dict, hotspot: dict) -> str:
    """Claude-generated Turkish description. Falls back to a sensible default."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    show    = _strip_episode(job.get("title", "Dizi"))

    if not api_key:
        return f"{show} — en çok tekrar izlenen an. #Shorts #Dizi"

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": (
                    f"Türk dizisi '{show}' için YouTube Shorts açıklaması yaz.\n"
                    "Max 120 karakter, Türkçe, spoiler verme, merak uyandır.\n"
                    "Sadece açıklama metnini yaz, başka hiçbir şey yazma."
                ),
            }],
        )
        return message.content[0].text.strip()
    except Exception as e:
        log.warning(f"Claude description failed: {e}")
        return f"{show} — en çok tekrar izlenen an. #Shorts #Dizi"

def build_tags(job: dict) -> list[str]:
    """Turkish-first tags derived from the show title."""
    show  = _strip_episode(job.get("title", "Dizi"))
    slug  = show.replace(" ", "")          # e.g. "KızılcıkŞerbeti"
    words = [w for w in show.split() if len(w) > 2][:4]
    # full show name first (most searchable), then slug, individual words, generic tags
    return [show, slug] + words + ["shorts", "dizi", "türkdizisi", "türkdizileri"]


# =============================================================================
# Upload — reuses same credentials as existing youtube_upload.py
# =============================================================================

def upload_to_youtube(clip_path: Path, title: str, description: str, tags: list[str]) -> str | None:
    """
    Reuses the same upload path as agents/youtube_upload.py.
    Imports it directly to avoid duplicating OAuth logic.
    """
    try:
        from agents.youtube_upload import upload_to_youtube as _upload
        video_id = _upload(clip_path, title, description, tags)
        return video_id
    except ImportError:
        pass

    # Fallback: inline upload using googleapiclient
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        log.error("google-api-python-client not installed. Run: pip install google-api-python-client")
        return None

    client_id     = settings.YOUTUBE_CLIENT_ID
    client_secret = settings.YOUTUBE_CLIENT_SECRET
    refresh_token = settings.YOUTUBE_REFRESH_TOKEN

    if not all([client_id, client_secret, refresh_token]):
        log.error("Missing YouTube credentials in settings")
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
        "snippet": {
            "title":           title,
            "description":     description,
            "tags":            tags,
            "categoryId":      "24",   # Entertainment
            "defaultLanguage": "tr",
        },
        "status": {
            "privacyStatus":           os.getenv("YT_UPLOAD_PRIVACY", "public"),
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        str(clip_path),
        mimetype="video/mp4",
        resumable=True,
        chunksize=10 * 1024 * 1024,
    )

    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            log.info(f"Upload {int(status.progress() * 100)}%...")

    video_id = response["id"]
    log.info(f"Uploaded: https://youtube.com/shorts/{video_id}")
    return video_id


# =============================================================================
# Main entry point — called by discord_bot.py on ✅ reaction
# =============================================================================

def handle_approval(record: dict) -> str | None:
    """
    Called from discord_bot.py's on_raw_reaction_add when type == 'soap_short'.
    record keys: clip_path, job, hotspot, clip_index
    Returns YouTube Shorts URL or None.
    """
    clip_path = Path(record["clip_path"])
    job       = record["job"]
    hotspot   = record["hotspot"]

    # clip_index: handle both 0-based and 1-based
    raw_index   = record.get("clip_index", 0)
    clip_number = raw_index if raw_index >= 1 else raw_index + 1

    if not clip_path.exists():
        log.error(f"Clip file not found: {clip_path}")
        return None

    title       = build_title(job, hotspot, clip_index=clip_number)
    description = build_description(job, hotspot)
    tags        = build_tags(job)

    log.info(f"Uploading: {title}")
    video_id = upload_to_youtube(clip_path, title, description, tags)

    if video_id:
        clip_path.unlink(missing_ok=True)   # clean up after successful upload
        return f"https://youtube.com/shorts/{video_id}"
    return None


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Soap Opera Shorts — YouTube Uploader")
    parser.add_argument("--record", required=True, help="JSON record string from soap_discord_pending.jsonl")
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
