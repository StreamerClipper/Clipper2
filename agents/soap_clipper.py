"""
agents/soap_clipper.py — Soap Opera Shorts: Clipper

Runs inside GitHub Actions (triggered by soap_pending.jsonl commit).
For each queued URL:
  1. Fetches metadata + Most Replayed heatmap via yt-dlp
  2. Detects top 3 non-overlapping 30s hotspots
  3. Downloads each segment with yt-dlp --download-sections
  4. Crops to 9:16 with ffmpeg
  5. Burns Turkish/English subtitles (with timestamp offset)
  6. Posts each clip to Discord for approval
"""
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
import html

import requests

from config.settings import settings

log = logging.getLogger("soap_clipper")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

SOAP_PENDING_FILE   = Path("output/soap_pending.jsonl")
SOAP_PROCESSED_FILE = Path("output/soap_processed.jsonl")
SOAP_DISCORD_FILE   = Path("output/soap_discord_pending.jsonl")
CLIPS_DIR           = Path("output/clips")
TMP_DIR             = Path("/tmp/soap_clipper")

CLIP_DURATION = 45
TOP_N         = 3
OUT_W, OUT_H  = 608, 1080

DISCORD_BOT_TOKEN = settings.DISCORD_BOT_TOKEN

# Soap-specific Discord channels
SOAP_CLIPS_CHANNEL_ID = "1484834736257106020"
SOAP_LOG_CHANNEL_ID   = "1484834748181385256"
SOAP_INPUT_CHANNEL_ID = "1484842601617293394"


# =============================================================================
# Discord logging
# =============================================================================

def discord_log(message: str, channel_id: str = None):
    if not DISCORD_BOT_TOKEN:
        return
    cid = channel_id or SOAP_LOG_CHANNEL_ID
    try:
        requests.post(
            f"https://discord.com/api/v10/channels/{cid}/messages",
            headers={"Authorization": f"Bot {DISCORD_BOT_TOKEN}"},
            json={"content": message},
            timeout=5,
        )
    except Exception:
        pass


# =============================================================================
# Helpers
# =============================================================================

def ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def ts_label(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# =============================================================================
# Queue management
# =============================================================================

def load_next_job():
    # Primary: read from soap_pending.jsonl
    if SOAP_PENDING_FILE.exists():
        lines = [l for l in SOAP_PENDING_FILE.read_text().strip().splitlines() if l.strip()]
        if lines:
            return json.loads(lines[-1]), lines

    # Fallback: read from .soap_trigger (written by scout, committed to git)
    trigger = Path(".soap_trigger")
    if trigger.exists():
        try:
            data = json.loads(trigger.read_text().strip())
            url = data.get("url", "")
            if url:
                job = {
                    "url":       url,
                    "title":     url,
                    "queued_at": data.get("queued_at", ""),
                    "mute":      data.get("mute", False),  # ← add this
                }
                log.info(f"Loaded job from .soap_trigger: {url}")
                return job, []
        except Exception as e:
            log.error(f"Failed to read .soap_trigger: {e}")

    return None, None


def mark_processed(job: dict, lines: list[str]):
    remaining = lines[:-1]
    SOAP_PENDING_FILE.write_text("\n".join(remaining) + ("\n" if remaining else ""))
    job["processed_at"] = datetime.now(timezone.utc).isoformat()
    with open(SOAP_PROCESSED_FILE, "a") as f:
        f.write(json.dumps(job) + "\n")


# =============================================================================
# Step 1 — Fetch metadata + heatmap
# =============================================================================

def fetch_video_metadata(url: str) -> dict | None:
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-playlist",
        "--no-warnings",
        "--cookies", "cookies.txt",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.error(f"yt-dlp metadata fetch failed: {result.stderr[:400]}")
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        log.error("yt-dlp returned invalid JSON")
        return None

    raw_heatmap = data.get("heatmap") or []
    heatmap = [
        {
            "start":     p.get("start_time", 0),
            "end":       p.get("end_time", 0),
            "intensity": p.get("value", 0.0),
        }
        for p in raw_heatmap
    ]

    return {
        "video_id":    data["id"],
        "title":       data.get("title", ""),
        "url":         f"https://www.youtube.com/watch?v={data['id']}",
        "duration":    data.get("duration", 0),
        "heatmap":     heatmap,
        "upload_date": data.get("upload_date", ""),
    }


# =============================================================================
# Step 2 — Hotspot detection
# =============================================================================

def find_hotspots(heatmap: list[dict]) -> list[dict]:
    if not heatmap:
        return []

    half   = CLIP_DURATION / 2
    ranked = sorted(heatmap, key=lambda p: p["intensity"], reverse=True)
    chosen  = []
    windows = []

    for point in ranked:
        if len(chosen) >= TOP_N:
            break

        peak_sec  = (point["start"] + point["end"]) / 2
        win_start = max(0.0, peak_sec - half)
        win_end   = win_start + CLIP_DURATION

        if any(not (win_end <= ws or win_start >= we) for ws, we in windows):
            continue

        chosen.append({
            "start_sec": win_start,
            "peak_sec":  peak_sec,
            "end_sec":   win_end,
            "intensity": point["intensity"],
        })
        windows.append((win_start, win_end))

    return sorted(chosen, key=lambda h: h["start_sec"])


# =============================================================================
# Step 3 — Download hotspot segment
# =============================================================================

def download_segment(url: str, start: float, duration: int, out: Path) -> bool:
    section = f"*{ts(start)}-{ts(start + duration)}"
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "--cookies", "cookies.txt",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--download-sections", section,
        "--force-keyframes-at-cuts",
        "-o", str(out),
        url,
    ]
    log.info(f"Downloading {ts_label(start)}–{ts_label(start+duration)}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not out.exists():
        log.error(f"yt-dlp download failed: {result.stderr[:400]}")
        return False
    log.info(f"Downloaded: {out.stat().st_size/1024/1024:.1f}MB")
    return True


# =============================================================================
# Step 4 — Fetch subtitles
# =============================================================================

def fetch_subtitles(url: str, stem: Path) -> Path | None:
    # Try official subtitles first, then auto-generated
    for sub_type in ["--write-sub", "--write-auto-sub"]:
        for lang in ("tr", "en", "en-US"):
            cmd = [
                "yt-dlp",
                "--no-playlist",
                "--skip-download",
                "--no-warnings",
                "--cookies", "cookies.txt",
                sub_type,
                "--sub-lang", lang,
                "--sub-format", "vtt",
                "-o", str(stem),
                url,
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            candidate = stem.parent / f"{stem.name}.{lang}.vtt"
            if candidate.exists():
                log.info(f"Subtitles fetched: {lang} ({'official' if sub_type == '--write-sub' else 'auto-generated'})")
                return candidate

    log.warning("No subtitles available")
    return None


# =============================================================================
# Step 4b — Shift subtitle timestamps
# =============================================================================

def vtt_time_to_seconds(t: str) -> float:
    """Convert VTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) to seconds."""
    parts = t.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
    else:
        h, m, s = 0, parts[0], parts[1]
    return int(h) * 3600 + int(m) * 60 + float(s)


def seconds_to_srt_time(s: float) -> str:
    """Convert seconds to SRT timestamp HH:MM:SS,mmm"""
    s = max(0.0, s)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    ms = int((sec % 1) * 1000)
    return f"{h:02d}:{m:02d}:{int(sec):02d},{ms:03d}"


def shift_subtitles_to_srt(vtt_path: Path, start_sec: float, out_srt: Path) -> bool:
    """
    Convert VTT to SRT and shift all timestamps back by start_sec.
    This aligns subtitles with the clipped segment (which starts at 0s).
    """
    try:
        text = vtt_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()

        # Regex to match VTT timestamp lines: 00:05:30.000 --> 00:05:32.500
        ts_pattern = re.compile(
            r"(\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{2}:\d{2}[.,]\d{3})"
            r"\s*-->\s*"
            r"(\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{2}:\d{2}[.,]\d{3})"
        )

        srt_entries = []
        i = 0
        counter = 1

        while i < len(lines):
            line = lines[i].strip()
            match = ts_pattern.match(line)
            if match:
                # Parse timestamps
                start_ts = match.group(1).replace(",", ".")
                end_ts   = match.group(2).replace(",", ".")
                start_s  = vtt_time_to_seconds(start_ts) - start_sec
                end_s    = vtt_time_to_seconds(end_ts) - start_sec

                # Skip subtitles outside our clip window
                if end_s < 0 or start_s > CLIP_DURATION:
                    i += 1
                    continue

                # Clamp to clip bounds
                start_s = max(0.0, start_s)
                end_s   = min(float(CLIP_DURATION), end_s)

                # Collect subtitle text lines
                i += 1
                sub_lines = []
                while i < len(lines) and lines[i].strip() != "" and not ts_pattern.match(lines[i].strip()):
                    txt = html.unescape(lines[i].strip())
                    # Skip VTT cue settings and WEBVTT header
                    if txt and not txt.startswith("WEBVTT") and not txt.startswith("NOTE") and "<" not in txt:
                        sub_lines.append(txt)
                    i += 1

                if sub_lines:
                    srt_entries.append(
                        f"{counter}\n"
                        f"{seconds_to_srt_time(start_s)} --> {seconds_to_srt_time(end_s)}\n"
                        f"{chr(10).join(sub_lines)}\n"
                    )
                    counter += 1
            else:
                i += 1

        if not srt_entries:
            log.warning("No subtitle entries in clip window — skipping subtitles")
            return False

        out_srt.write_text("\n".join(srt_entries), encoding="utf-8")
        log.info(f"Shifted {len(srt_entries)} subtitle entries (offset: -{start_sec:.1f}s) → {out_srt.name}")
        return True

    except Exception as e:
        log.error(f"Subtitle shift failed: {e}")
        return False


def mute_audio(input_path: Path, output_path: Path) -> bool:
    """Strip audio track to avoid copyright claims."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-an",
        "-c:v", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        log.warning(f"Mute failed — keeping audio: {result.stderr[-200:]}")
        shutil.copy(input_path, output_path)
    return True

# =============================================================================
# Step 5 — Crop to 9:16
# =============================================================================

def get_video_dimensions(path: Path) -> tuple[int, int]:
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    video = next((s for s in streams if s["codec_type"] == "video"), None)
    if not video:
        return 1920, 1080
    return int(video["width"]), int(video["height"])


def crop_to_vertical(input_path: Path, output_path: Path) -> bool:
    w, h = get_video_dimensions(input_path)
    target_w = int(h * 9 / 16)
    crop_x   = (w - target_w) // 2

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"crop={target_w}:{h}:{crop_x}:0,scale={OUT_W}:{OUT_H}",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.error(f"ffmpeg crop failed: {result.stderr[-400:]}")
        return False
    log.info(f"Cropped to 9:16: {output_path}")
    return True


# =============================================================================
# Step 6 — Burn subtitles (using shifted SRT)
# =============================================================================

def burn_subtitles(input_path: Path, srt_path: Path, output_path: Path) -> bool:
    safe_sub = str(srt_path).replace("\\", "/").replace(":", "\\:")
    vf = (
        f"subtitles='{safe_sub}'"
        f":force_style='FontSize=14,Bold=1,"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"Outline=2,Shadow=1,MarginV=30'"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-preset", "fast",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.warning(f"Subtitle burn failed — using clip without subtitles: {result.stderr[-300:]}")
        shutil.copy(input_path, output_path)
    return True


# =============================================================================
# Step 7 — Compress if too large for Discord (8MB limit)
# =============================================================================

def compress_for_discord(clip_path: Path) -> Path:
    size_mb = clip_path.stat().st_size / 1024 / 1024
    if size_mb <= 24:
        return clip_path

    log.warning(f"Clip too large ({size_mb:.1f}MB) — compressing...")
    compressed = clip_path.with_suffix(".small.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(clip_path),
        "-c:v", "libx264", "-crf", "28",
        "-c:a", "aac", "-b:a", "96k",
        "-preset", "fast",
        str(compressed),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode == 0 and compressed.exists():
        log.info(f"Compressed: {compressed.stat().st_size/1024/1024:.1f}MB")
        return compressed
    return clip_path


# =============================================================================
# Step 8 — Post to Discord for approval
# =============================================================================

def send_clip_to_discord(clip_path: Path, job: dict, hotspot: dict, clip_index: int) -> str | None:
    if not DISCORD_BOT_TOKEN:
        log.error("DISCORD_BOT_TOKEN not set")
        return None

    clip_path = compress_for_discord(clip_path)

    intensity_pct = int(hotspot["intensity"] * 100)
    content = (
        f"📺 **[Soap Shorts]** Clip {clip_index+1}/3 ready for approval\n\n"
        f"**Episode:** {job['title']}\n"
        f"**Timestamp:** `{ts_label(hotspot['start_sec'])}` — `{ts_label(hotspot['end_sec'])}`\n"
        f"**Most Replayed intensity:** `{intensity_pct}%`\n\n"
        f"React with ✅ to upload to YouTube Shorts, or ❌ to discard."
    )

    try:
        url = f"https://discord.com/api/v10/channels/{SOAP_CLIPS_CHANNEL_ID}/messages"
        headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}

        with open(clip_path, "rb") as f:
            resp = requests.post(
                url,
                headers=headers,
                files={"file": (clip_path.name, f, "video/mp4")},
                data={"content": content},
                timeout=120,
            )

        if resp.status_code not in (200, 201):
            log.error(f"Discord upload failed ({resp.status_code}): {resp.text[:300]}")
            return None

        message_id = resp.json()["id"]
        log.info(f"Clip {clip_index+1} sent to Discord — message {message_id}")

        import time
        for emoji in ["✅", "❌"]:
            encoded = urllib.parse.quote(emoji)
            react_url = f"https://discord.com/api/v10/channels/{SOAP_CLIPS_CHANNEL_ID}/messages/{message_id}/reactions/{encoded}/@me"
            requests.put(react_url, headers=headers, timeout=5)
            time.sleep(0.5)

        record = {
            "message_id": message_id,
            "clip_path":  str(clip_path),
            "job":        {k: v for k, v in job.items() if k != "heatmap"},
            "hotspot":    hotspot,
            "clip_index": clip_index,
            "type":       "soap_short",
        }
        SOAP_DISCORD_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SOAP_DISCORD_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

        return message_id

    except Exception as e:
        log.error(f"Discord send failed: {e}")
        return None

def generate_whisper_subtitles(video_path: Path, stem: Path) -> Path | None:
    """
    Generate subtitles using OpenAI Whisper when no official subs available.
    """
    try:
        import subprocess
        subprocess.run(
            ["pip", "install", "openai-whisper", "--quiet"],
            capture_output=True, timeout=120
        )
        cmd = [
            "whisper",
            str(video_path),
            "--language", "tr",
            "--output_format", "vtt",
            "--output_dir", str(stem.parent),
            "--model", "medium",
            "--task", "transcribe",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Whisper names output after the input file, not our stem
        # e.g. input: soap_xxx_raw.mp4 → output: soap_xxx_raw.vtt
        expected = stem.parent / f"{video_path.stem}.vtt"
        out_vtt  = Path(f"{stem}.vtt")

        if expected.exists():
            expected.rename(out_vtt)
            log.info(f"Whisper generated subtitles: {out_vtt}")
            return out_vtt
        else:
            # Search for any .vtt in the output dir as fallback
            vtts = list(stem.parent.glob("*.vtt"))
            if vtts:
                vtts[0].rename(out_vtt)
                log.info(f"Whisper generated subtitles (found): {out_vtt}")
                return out_vtt
            log.warning(f"Whisper failed: {result.stderr[-200:]}")
            return None
    except Exception as e:
        log.error(f"Whisper subtitle generation failed: {e}")
        return None

# =============================================================================
# Process one hotspot
# =============================================================================

def process_hotspot(job: dict, hotspot: dict, clip_index: int) -> Path | None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    vid   = job["video_id"]
    slug  = f"soap_{vid}_clip{clip_index}_{datetime.now(timezone.utc).strftime('%H%M%S')}"
    url   = job["url"]
    start = hotspot["start_sec"]
    mute    = job.get("mute", False)

    raw     = TMP_DIR / f"{slug}_raw.mp4"
    cropped = TMP_DIR / f"{slug}_cropped.mp4"
    final   = CLIPS_DIR / f"{slug}_final.mp4"

    if not download_segment(url, start, CLIP_DURATION, raw):
        return None

    if not crop_to_vertical(raw, cropped):
        return None
    raw.unlink(missing_ok=True)

    # Fetch subtitles once per job, reuse across clips
    vtt_path = TMP_DIR / f"soap_{vid}_subs.vtt"
    if not vtt_path.exists():
        fetched = fetch_subtitles(url, TMP_DIR / f"soap_{vid}_subs")
        if fetched:
            fetched.rename(vtt_path)
        elif mute:
            # No subtitles available + muted — generate with Whisper
            log.info("No subtitles found — generating with Whisper...")
            generated = generate_whisper_subtitles(raw if raw.exists() else cropped, TMP_DIR / f"soap_{vid}_subs")
            if generated:
                generated.rename(vtt_path)

    if mute:
        muted = TMP_DIR / f"{slug}_muted.mp4"
        mute_audio(cropped, muted)
        cropped.unlink(missing_ok=True)
        cropped = muted

    # Shift subtitle timestamps to match clip window
    if vtt_path.exists():
        srt_path = TMP_DIR / f"{slug}_shifted.srt"
        has_subs = shift_subtitles_to_srt(vtt_path, max(0.0, start), srt_path)
        if has_subs:
            subbed = TMP_DIR / f"{slug}_subbed.mp4"
            burn_subtitles(cropped, srt_path, subbed)
            cropped.unlink(missing_ok=True)
            srt_path.unlink(missing_ok=True)
            shutil.move(str(subbed), str(final))
        else:
            shutil.move(str(cropped), str(final))
    else:
        shutil.move(str(cropped), str(final))

    log.info(f"Clip ready: {final}")
    return final


# =============================================================================
# Main
# =============================================================================

def main():
    job, lines = load_next_job()
    if job is None:
        log.info("No pending soap jobs — nothing to do")
        sys.exit(0)

    url = job["url"]
    log.info(f"Fetching metadata for: {url}")

    meta = fetch_video_metadata(url)
    if not meta:
        discord_log(f"❌ **[Soap]** Could not fetch metadata for `{url}`", channel_id=SOAP_LOG_CHANNEL_ID)
        mark_processed(job, lines)
        sys.exit(1)

    if not meta["heatmap"]:
        discord_log(
            f"⚠️ **[Soap]** No Most Replayed data for *{meta['title']}*\n"
            f"The video may be too new or have too few views.",
            channel_id=SOAP_LOG_CHANNEL_ID,
        )
        mark_processed(job, lines)
        sys.exit(0)

    hotspots = find_hotspots(meta["heatmap"])
    if not hotspots:
        discord_log(f"⚠️ **[Soap]** No usable hotspots in *{meta['title']}*", channel_id=SOAP_LOG_CHANNEL_ID)
        mark_processed(job, lines)
        sys.exit(0)

    job.update(meta)
    job["hotspots"] = hotspots

    hs_summary = "\n".join(
        f"  • Clip {i+1}: `{ts_label(h['start_sec'])}` — `{h['intensity']:.0%}` intensity"
        for i, h in enumerate(hotspots)
    )
    discord_log(
        f"⚙️ **[Soap]** Processing *{meta['title']}*\n"
        f"Hotspots:\n{hs_summary}",
        channel_id=SOAP_LOG_CHANNEL_ID,
    )

    clips_sent = 0
    for i, hotspot in enumerate(hotspots):
        log.info(f"--- Hotspot {i+1}/{len(hotspots)} @ {ts_label(hotspot['start_sec'])} ---")
        clip_path = process_hotspot(job, hotspot, i)
        if clip_path:
            msg_id = send_clip_to_discord(clip_path, job, hotspot, i)
            if msg_id:
                clips_sent += 1
        else:
            log.warning(f"Hotspot {i+1} failed — skipping")

    mark_processed(job, lines)
    discord_log(
        f"✅ **[Soap]** Done — {clips_sent}/{len(hotspots)} clips sent for approval.",
        channel_id=SOAP_LOG_CHANNEL_ID,
    )


if __name__ == "__main__":
    main()