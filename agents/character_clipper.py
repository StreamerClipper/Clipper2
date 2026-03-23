"""
agents/character_clipper.py — Character Highlight Reel Generator

Triggered by .character_trigger file commit.
For a given episode URL + character name:
  1. Downloads full episode
  2. Extracts keyframes (1 per second)
  3. Uses insightface to find frames matching reference face
  4. Merges matching segments into 3-5 min highlight video
  5. Applies same transforms as soap_clipper (mirror, zoom, speed, music, subs)
  6. Posts to Discord for approval → uploads to dizikliper as regular video
"""
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

from config.settings import settings

log = logging.getLogger("character_clipper")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

CLIPS_DIR   = Path("output/clips")
TMP_DIR     = Path("/tmp/character_clipper")
FACES_DIR   = Path("faces")

MIN_DURATION = 180   # 3 minutes
MAX_DURATION = 300   # 5 minutes
KEYFRAME_INTERVAL = 1    # seconds between keyframes
MIN_SCENE_GAP     = 3    # seconds gap = new scene
MIN_SCENE_LENGTH  = 5    # minimum scene length to include
FACE_THRESHOLD    = 0.45 # cosine similarity threshold

DISCORD_BOT_TOKEN     = settings.DISCORD_BOT_TOKEN
SOAP_CLIPS_CHANNEL_ID = "1484834736257106020"
SOAP_LOG_CHANNEL_ID   = "1484834748181385256"

OUT_W, OUT_H = 608, 1080


# =============================================================================
# Discord logging
# =============================================================================

def discord_log(message: str):
    if not DISCORD_BOT_TOKEN:
        return
    try:
        requests.post(
            f"https://discord.com/api/v10/channels/{SOAP_LOG_CHANNEL_ID}/messages",
            headers={"Authorization": f"Bot {DISCORD_BOT_TOKEN}"},
            json={"content": message},
            timeout=5,
        )
    except Exception:
        pass


# =============================================================================
# Trigger file
# =============================================================================

def load_job():
    trigger = Path(".character_trigger")
    if not trigger.exists():
        return None
    try:
        data = json.loads(trigger.read_text().strip())
        return {
            "url":       data["url"],
            "character": data["character"],   # e.g. "doga"
            "title":     data.get("title", data["url"]),
            "queued_at": data.get("queued_at", ""),
        }
    except Exception as e:
        log.error(f"Failed to read .character_trigger: {e}")
        return None


# =============================================================================
# Step 1 — Download full episode
# =============================================================================

def download_episode(url: str, out: Path) -> bool:
    cmd = [
        "yt-dlp",
        "--no-playlist", "--no-warnings",
        "--cookies", "cookies.txt",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "-o", str(out),
        url,
    ]
    log.info(f"Downloading full episode...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0 or not out.exists():
        log.error(f"Download failed: {result.stderr[:400]}")
        return False
    log.info(f"Downloaded: {out.stat().st_size/1024/1024:.0f}MB")
    return True


# =============================================================================
# Step 2 — Face recognition setup
# =============================================================================

def load_face_model():
    """Load insightface model for face embeddings."""
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(320, 320))
        log.info("insightface model loaded")
        return app
    except Exception as e:
        log.error(f"Failed to load face model: {e}")
        return None


def get_face_embedding(app, image_path: Path) -> np.ndarray | None:
    """Get face embedding from image."""
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        faces = app.get(img)
        if not faces:
            return None
        # Return embedding of largest face
        largest = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        return largest.normed_embedding
    except Exception as e:
        log.warning(f"Embedding failed for {image_path}: {e}")
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# =============================================================================
# Step 3 — Scan episode for character appearances
# =============================================================================

def scan_episode(episode_path: Path, ref_embedding: np.ndarray, app) -> list[float]:
    """
    Extract keyframes every KEYFRAME_INTERVAL seconds and find
    timestamps where the target character appears.
    Returns list of matching timestamps in seconds.
    """
    import cv2

    cap = cv2.VideoCapture(str(episode_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    log.info(f"Scanning {duration:.0f}s episode at {fps:.1f}fps...")

    frame_interval = int(fps * KEYFRAME_INTERVAL)
    matching_timestamps = []
    frame_num = 0
    scanned = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        # Run face detection
        faces = app.get(frame)
        for face in faces:
            sim = cosine_similarity(ref_embedding, face.normed_embedding)
            if sim >= FACE_THRESHOLD:
                timestamp = frame_num / fps
                matching_timestamps.append(timestamp)
                log.info(f"  Match at {timestamp:.1f}s (similarity={sim:.3f})")
                break

        frame_num += frame_interval
        scanned += 1
        if scanned % 60 == 0:
            log.info(f"  Scanned {frame_num/fps:.0f}s / {duration:.0f}s ({100*frame_num/total_frames:.0f}%)")

    cap.release()
    log.info(f"Found {len(matching_timestamps)} matching timestamps")
    return matching_timestamps


# =============================================================================
# Step 4 — Build scene list from timestamps
# =============================================================================

def timestamps_to_scenes(timestamps: list[float], episode_duration: float) -> list[dict]:
    """
    Merge nearby timestamps into scenes with start/end times.
    """
    if not timestamps:
        return []

    scenes = []
    scene_start = timestamps[0]
    scene_end   = timestamps[0] + KEYFRAME_INTERVAL

    for t in timestamps[1:]:
        if t - scene_end <= MIN_SCENE_GAP:
            # Extend current scene
            scene_end = t + KEYFRAME_INTERVAL
        else:
            # New scene
            if scene_end - scene_start >= MIN_SCENE_LENGTH:
                scenes.append({
                    "start": max(0, scene_start - 1),  # 1s padding
                    "end":   min(episode_duration, scene_end + 1),
                })
            scene_start = t
            scene_end   = t + KEYFRAME_INTERVAL

    # Last scene
    if scene_end - scene_start >= MIN_SCENE_LENGTH:
        scenes.append({
            "start": max(0, scene_start - 1),
            "end":   min(episode_duration, scene_end + 1),
        })

    log.info(f"Built {len(scenes)} scenes, total={sum(s['end']-s['start'] for s in scenes):.0f}s")
    return scenes


def select_scenes(scenes: list[dict]) -> list[dict]:
    """Select scenes to fill 3-5 minutes."""
    total = 0
    selected = []
    for scene in scenes:
        duration = scene["end"] - scene["start"]
        if total + duration > MAX_DURATION:
            # Trim last scene to fit
            remaining = MAX_DURATION - total
            if remaining >= MIN_SCENE_LENGTH:
                scene = dict(scene)
                scene["end"] = scene["start"] + remaining
                selected.append(scene)
            break
        selected.append(scene)
        total += duration
        if total >= MIN_DURATION:
            break

    log.info(f"Selected {len(selected)} scenes, total={sum(s['end']-s['start'] for s in selected):.0f}s")
    return selected


# =============================================================================
# Step 5 — Download and assemble scenes
# =============================================================================

def ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def download_scene(url: str, scene: dict, idx: int, out: Path) -> bool:
    section = f"*{ts(scene['start'])}-{ts(scene['end'])}"
    cmd = [
        "yt-dlp", "--no-playlist", "--no-warnings",
        "--cookies", "cookies.txt",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--download-sections", section,
        "--force-keyframes-at-cuts",
        "-o", str(out),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0 or not out.exists():
        log.warning(f"Scene {idx} download failed")
        return False
    return True


def assemble_scenes(scene_paths: list[Path], out: Path) -> bool:
    """Concatenate scenes into one video."""
    concat_file = TMP_DIR / "concat.txt"
    concat_file.write_text("\n".join(f"file '{p}'" for p in scene_paths))

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    concat_file.unlink(missing_ok=True)
    return result.returncode == 0 and out.exists()


# =============================================================================
# Step 6 — Apply transforms (reuse soap_clipper logic)
# =============================================================================

def crop_to_vertical(input_path: Path, output_path: Path) -> bool:
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(input_path)],
        capture_output=True, text=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    video = next((s for s in streams if s["codec_type"] == "video"), None)
    if not video:
        return False
    w, h = int(video["width"]), int(video["height"])
    target_w = int(h * 9 / 16)
    crop_x = (w - target_w) // 2

    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", f"crop={target_w}:{h}:{crop_x}:0,scale={OUT_W}:{OUT_H}",
        "-c:v", "libx264", "-c:a", "aac", "-preset", "fast",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return result.returncode == 0


def transform_video(input_path: Path, output_path: Path) -> bool:
    """Apply Content ID bypass transforms — same as soap_clipper."""
    cta_path   = Path(__file__).parent.parent / "abone_ol.mp4"
    music_path = Path(__file__).parent.parent / "drama_sfx.mp3"

    video_filter = (
        "hflip,"
        "zoompan=z=1.04:d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=608x1080,"
        "eq=saturation=1.15:brightness=0.02,"
        "vignette=PI/4,"
        "setpts=PTS/1.2"
    )
    audio_filter = "atempo=1.2,aecho=0.8:0.88:60:0.1,asetrate=44100*1.03,aresample=44100"

    if cta_path.exists():
        filter_complex = (
            f"[0:v]{video_filter}[transformed];"
            f"[1:v]scale=608:280,fade=t=in:st=0:d=0.2:alpha=0,"
            f"fade=t=out:st=2.8:d=0.4:alpha=0,"
            f"colorkey=black:0.12:0.03[cta];"
            f"[transformed][cta]overlay=0:50:enable='between(t,0,3.5)'[outv]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path), "-i", str(cta_path),
            "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "0:a",
            "-af", audio_filter,
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vf", video_filter, "-af", audio_filter,
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k",
            str(output_path),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        log.warning(f"Transform failed: {result.stderr[-200:]}")
        shutil.copy(input_path, output_path)

    # Mix background music
    if music_path.exists():
        music_out = output_path.with_suffix('.music.mp4')
        music_cmd = [
            "ffmpeg", "-y",
            "-i", str(output_path),
            "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex",
            "[0:a]volume=1.0[dialogue];"
            "[1:a]volume=0.35,afade=t=in:st=0:d=2[music];"
            "[dialogue][music]amix=inputs=2:duration=first[outa]",
            "-map", "0:v", "-map", "[outa]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
            str(music_out),
        ]
        res = subprocess.run(music_cmd, capture_output=True, text=True, timeout=600)
        if res.returncode == 0:
            output_path.unlink(missing_ok=True)
            music_out.rename(output_path)

    return True


# =============================================================================
# Step 7 — Post to Discord
# =============================================================================

def send_to_discord(clip_path: Path, job: dict, total_duration: float) -> str | None:
    if not DISCORD_BOT_TOKEN:
        return None

    import urllib.parse, time

    record = {
        "clip_path":  str(clip_path),
        "job": {
            "url":       job["url"],
            "title":     job["title"],
            "character": job["character"],
        },
        "type": "character_highlight",
    }

    content = (
        f"🎬 **[Character Highlight]** Ready for approval\n\n"
        f"**Episode:** {job['title']}\n"
        f"**Character:** `{job['character']}`\n"
        f"**Duration:** `{total_duration/60:.1f} min`\n\n"
        f"React with ✅ to upload to YouTube, or ❌ to discard.\n"
        f"||`RECORD:{json.dumps(record, separators=(',', ':'))}`||"
    )

    try:
        url = f"https://discord.com/api/v10/channels/{SOAP_CLIPS_CHANNEL_ID}/messages"
        headers = {"Authorization": f"Bot {DISCORD_BOT_TOKEN}"}
        with open(clip_path, "rb") as f:
            resp = requests.post(
                url, headers=headers,
                files={"file": (clip_path.name, f, "video/mp4")},
                data={"content": content},
                timeout=300,
            )
        if resp.status_code not in (200, 201):
            log.error(f"Discord upload failed: {resp.text[:300]}")
            return None

        message_id = resp.json()["id"]
        for emoji in ["✅", "❌"]:
            encoded = urllib.parse.quote(emoji)
            react_url = f"https://discord.com/api/v10/channels/{SOAP_CLIPS_CHANNEL_ID}/messages/{message_id}/reactions/{encoded}/@me"
            requests.put(react_url, headers=headers, timeout=5)
            time.sleep(0.5)

        return message_id
    except Exception as e:
        log.error(f"Discord send failed: {e}")
        return None


# =============================================================================
# Main
# =============================================================================

def main():
    job = load_job()
    if not job:
        log.info("No character job — nothing to do")
        sys.exit(0)

    character = job["character"]
    url       = job["url"]
    ref_photo = FACES_DIR / f"{character}.jpg"

    if not ref_photo.exists():
        log.error(f"Reference photo not found: {ref_photo}")
        discord_log(f"❌ **[Character]** Reference photo not found: `faces/{character}.jpg`")
        sys.exit(1)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    slug    = f"char_{character}_{datetime.now(timezone.utc).strftime('%H%M%S')}"
    episode = TMP_DIR / f"{slug}_episode.mp4"
    final   = CLIPS_DIR / f"{slug}_final.mp4"

    discord_log(f"⚙️ **[Character]** Starting highlight reel for `{character}` from *{job['title']}*")

    # Install insightface if needed
    subprocess.run(["pip", "install", "insightface", "onnxruntime", "--quiet"],
                   capture_output=True, timeout=120)

    # Load face model
    app = load_face_model()
    if not app:
        discord_log(f"❌ **[Character]** Failed to load face recognition model")
        sys.exit(1)

    # Get reference embedding
    ref_embedding = get_face_embedding(app, ref_photo)
    if ref_embedding is None:
        log.error("Could not get embedding from reference photo")
        discord_log(f"❌ **[Character]** Could not detect face in `faces/{character}.jpg`")
        sys.exit(1)
    log.info(f"Reference embedding loaded for {character}")

    # Download full episode
    if not download_episode(url, episode):
        discord_log(f"❌ **[Character]** Episode download failed")
        sys.exit(1)

    # Get episode duration
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(episode)],
        capture_output=True, text=True
    )
    episode_duration = float(probe.stdout.strip() or 0)

    # Scan for character appearances
    matching_timestamps = scan_episode(episode, ref_embedding, app)

    if not matching_timestamps:
        discord_log(f"⚠️ **[Character]** No scenes found for `{character}` in *{job['title']}*")
        episode.unlink(missing_ok=True)
        sys.exit(0)

    # Build and select scenes
    scenes   = timestamps_to_scenes(matching_timestamps, episode_duration)
    selected = select_scenes(scenes)

    if not selected:
        discord_log(f"⚠️ **[Character]** Not enough scenes found (need {MIN_DURATION}s)")
        episode.unlink(missing_ok=True)
        sys.exit(0)

    total_duration = sum(s["end"] - s["start"] for s in selected)
    discord_log(
        f"✂️ **[Character]** Found {len(selected)} scenes "
        f"({total_duration/60:.1f} min) for `{character}`"
    )

    # Download each scene
    scene_paths = []
    for i, scene in enumerate(selected):
        scene_out = TMP_DIR / f"{slug}_scene{i}.mp4"
        log.info(f"Downloading scene {i+1}/{len(selected)}: {ts(scene['start'])} — {ts(scene['end'])}")
        if download_scene(url, scene, i, scene_out):
            scene_paths.append(scene_out)
        else:
            log.warning(f"Scene {i+1} failed — skipping")

    if not scene_paths:
        discord_log(f"❌ **[Character]** All scene downloads failed")
        sys.exit(1)

    # Assemble into one video
    assembled = TMP_DIR / f"{slug}_assembled.mp4"
    if not assemble_scenes(scene_paths, assembled):
        discord_log(f"❌ **[Character]** Scene assembly failed")
        sys.exit(1)

    # Crop to vertical
    cropped = TMP_DIR / f"{slug}_cropped.mp4"
    if not crop_to_vertical(assembled, cropped):
        log.warning("Crop failed — using assembled")
        shutil.copy(assembled, cropped)
    assembled.unlink(missing_ok=True)

    # Apply transforms
    transform_video(cropped, final)
    cropped.unlink(missing_ok=True)

    # Clean up scenes
    for p in scene_paths:
        p.unlink(missing_ok=True)
    episode.unlink(missing_ok=True)

    # Send to Discord
    msg_id = send_to_discord(final, job, total_duration)
    if msg_id:
        discord_log(f"✅ **[Character]** `{character}` highlight sent for approval ({total_duration/60:.1f} min)")
    else:
        discord_log(f"❌ **[Character]** Discord upload failed")


if __name__ == "__main__":
    main()
