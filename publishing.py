import os
import glob
import logging
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from config import Settings

logger = logging.getLogger("PAULA.Publishing")

def upload_youtube_rotated(settings: Settings, video_path: str, title: str, description: str, tags: list[str], is_short: bool = True) -> str:
    channel_tokens = sorted(glob.glob(f"tokens/{settings.channel_id}_token_*.json")) or sorted(glob.glob("tokens/*.json"))
    if not channel_tokens:
        raise RuntimeError(f"No se encontraron tokens OAuth para el canal '{settings.channel_name}' ({settings.channel_id}).")

    if settings.publish_at_iso and settings.publish_hour_art != "NOW":
        status_body = {
            "selfDeclaredMadeForKids": False,
            "privacyStatus": "private",
            "publishAt": settings.publish_at_iso
        }
    else:
        status_body = {
            "selfDeclaredMadeForKids": False,
            "privacyStatus": "public"
        }

    for token_path in channel_tokens:
        try:
            creds = Credentials.from_authorized_user_file(token_path, ["https://www.googleapis.com/auth/youtube.upload"])
            youtube = build("youtube", "v3", credentials=creds)
            body = {
                "snippet": {"title": title[:100], "description": description, "tags": tags, "categoryId": "27"},
                "status": status_body
            }
            media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
            resp = youtube.videos().insert(part="snippet,status", body=body, media_body=media).execute()
            vid_id = resp["id"]
            return f"https://youtube.com/shorts/{vid_id}" if is_short else f"https://youtu.be/{vid_id}"
        except HttpError as e:
            if e.resp.status in [403, 429]:
                continue
            raise e
    raise RuntimeError("Cuota de subida agotada en los tokens de YouTube disponibles.")
