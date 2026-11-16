import os
import glob
import json
import logging
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from config import Settings
from utils import retry_async

logger = logging.getLogger("PAULA.Publishing")

def upload_youtube_rotated(settings: Settings, video_path: str, title: str, description: str, tags: list[str], is_short: bool = True) -> str:
    token_path = f"tokens/{settings.channel_id}_token_kv.json"
    if not os.path.exists(token_path):
        # Fallback a cualquier token disponible en la carpeta
        all_tokens = sorted(glob.glob("tokens/*.json"))
        if all_tokens:
            token_path = all_tokens[0]
        else:
            raise RuntimeError(f"❌ No se encontró ningún archivo de token OAuth en tokens/ para el canal {settings.channel_name} ({settings.channel_id}).")

    logger.info(f"Usando credenciales OAuth desde: {token_path}")

    if settings.publish_at_iso and settings.publish_hour_art != "NOW":
        status_body = {
            "selfDeclaredMadeForKids": False,
            "privacyStatus": "private",
            "publishAt": settings.publish_at_iso
        }
        logger.info(f"Video configurado para publicación programada: {settings.publish_at_iso}")
    else:
        status_body = {
            "selfDeclaredMadeForKids": False,
            "privacyStatus": "public"
        }

    def _execute_upload():
        creds = Credentials.from_authorized_user_file(token_path, ["https://www.googleapis.com/auth/youtube.upload"])
        youtube = build("youtube", "v3", credentials=creds)
        body = {
            "snippet": {"title": title[:100], "description": description, "tags": tags, "categoryId": "27"},
            "status": status_body
        }
        media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
        resp = youtube.videos().insert(part="snippet,status", body=body, media_body=media).execute()
        return resp["id"]

    try:
        vid_id = asyncio_run_sync(retry_async(_execute_upload, retries=3, backoff=2.0))
        url = f"https://youtube.com/shorts/{vid_id}" if is_short else f"https://youtu.be/{vid_id}"
        logger.info(f"✅ Video subido con éxito a YouTube: {url}")
        return url
    except Exception as e:
        logger.error(f"❌ Error crítico subiendo a YouTube: {e}")
        raise e

def asyncio_run_sync(coro):
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)
