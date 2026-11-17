import os
import glob
import json
import logging
import httpx
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from config import Settings
from utils import retry_async

logger = logging.getLogger("PAULA.Publishing")

async def upload_youtube_rotated(
    settings: Settings,
    video_path: str,
    title: str,
    description: str,
    tags: list[str],
    is_short: bool = True
) -> str:
    os.makedirs("tokens", exist_ok=True)
    token_path = f"tokens/{settings.channel_id}_token_kv.json"

    if settings.channel_token_json:
        try:
            parsed = json.loads(settings.channel_token_json)
            full_token_data = {
                "token": parsed.get("token", "") or parsed.get("access_token", ""),
                "refresh_token": parsed.get("refresh_token", "") or parsed.get("refreshToken", ""),
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": parsed.get("client_id", "") or settings.google_client_id,
                "client_secret": parsed.get("client_secret", "") or settings.google_client_secret
            }
            with open(token_path, "w", encoding="utf-8") as f:
                json.dump(full_token_data, f)
            logger.info(f"Credenciales OAuth preparadas en {token_path}")
        except Exception as e:
            logger.warning(f"No se pudo reconstruir el token desde BUNDLE: {e}")

    if not os.path.exists(token_path):
        all_tokens = sorted(glob.glob("tokens/*.json"))
        if all_tokens:
            token_path = all_tokens[0]
        else:
            raise RuntimeError(f"❌ No se encontró token OAuth en tokens/ para el canal {settings.channel_name} ({settings.channel_id}).")

    if settings.publish_at_iso and settings.publish_hour_art != "NOW":
        status_body = {
            "selfDeclaredMadeForKids": False,
            "privacyStatus": "private",
            "publishAt": settings.publish_at_iso
        }
        logger.info(f"Programando estreno en YouTube para: {settings.publish_at_iso}")
    else:
        status_body = {
            "selfDeclaredMadeForKids": False,
            "privacyStatus": "public"
        }

    def _execute_upload() -> str:
        creds = Credentials.from_authorized_user_file(
            token_path,
            ["https://www.googleapis.com/auth/youtube.upload"]
        )
        if not creds.valid or not creds.token:
            creds.refresh(Request())

        youtube = build("youtube", "v3", credentials=creds)
        body = {
            "snippet": {
                "title": title[:100],
                "description": description,
                "tags": tags,
                "categoryId": "27"
            },
            "status": status_body
        }
        media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
        resp = youtube.videos().insert(part="snippet,status", body=body, media_body=media).execute()
        return resp["id"]

    vid_id = await retry_async(_execute_upload, retries=3, backoff=2.0)
    url = f"https://youtube.com/shorts/{vid_id}" if is_short else f"https://youtu.be/{vid_id}"
    logger.info(f"✅ Video publicado en YouTube: {url}")
    return url

async def distribute_omnichannel_short(settings: Settings, short_plan: dict, video_path: str, yt_short_url: str) -> list[str]:
    results = []

    if settings.platforms.get("x", True):
        x_text = short_plan.get("x_post") or f"{short_plan['title']} {yt_short_url}"
        if settings.x_bearer_token:
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    r = await client.post(
                        "https://api.twitter.com/2/tweets",
                        headers={"Authorization": f"Bearer {settings.x_bearer_token}", "Content-Type": "application/json"},
                        json={"text": x_text[:275]}
                    )
                    if r.status_code in [200, 201]:
                        tid = r.json().get("data", {}).get("id", "")
                        results.append(f"🐦 <b>X (Publicado):</b> https://x.com/i/web/status/{tid}")
                    else:
                        results.append(f"🐦 <b>X (Copy Listo):</b> <code>{x_text[:110]}...</code>")
            except Exception:
                results.append(f"🐦 <b>X (Copy Listo):</b> <code>{x_text[:110]}...</code>")
        else:
            results.append(f"🐦 <b>X (Post Listo):</b> <code>{x_text[:110]}...</code>")

    if settings.platforms.get("tiktok", True):
        results.append(f"🎵 <b>TikTok 9:16:</b> Masterizado (Caption: <code>{short_plan.get('tiktok_caption', '')[:65]}...</code>)")

    if settings.platforms.get("reels", True):
        results.append(f"📸 <b>IG Reels 9:16:</b> Masterizado (Caption: <code>{short_plan.get('reels_caption', '')[:65]}...</code>)")

    if settings.omnichannel_webhook_url:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                await client.post(settings.omnichannel_webhook_url, json={
                    "channel_id": settings.channel_id,
                    "channel_name": settings.channel_name,
                    "youtube_short_url": yt_short_url,
                    "title": short_plan["title"],
                    "tiktok_caption": short_plan.get("tiktok_caption", ""),
                    "reels_caption": short_plan.get("reels_caption", ""),
                    "x_post": short_plan.get("x_post", ""),
                    "publish_at_iso": settings.publish_at_iso,
                    "monetization_url": settings.monetization_url
                })
                results.append("🔗 <b>Webhook Omnicanal:</b> Despachado OK (TikTok/Reels/X)")
        except Exception as e:
            logger.warning(f"Webhook Omnicanal: {e}")

    return results
