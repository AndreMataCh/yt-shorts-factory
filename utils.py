import os
import re
import html
import json
import asyncio
import logging
import httpx
from typing import Callable, Any, Type
from config import Settings

logger = logging.getLogger("PAULA.Utils")

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}

API_HEADERS = {
    "User-Agent": "PAULA-Enterprise-Engine/8.0",
    "Accept": "application/json",
    "Content-Type": "application/json"
}

def format_ass_time(seconds: float) -> str:
    total_cs = int(round(max(0.0, float(seconds)) * 100))
    h = total_cs // 360000
    m = (total_cs % 360000) // 6000
    s = (total_cs % 6000) // 100
    cs = total_cs % 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def sanitize_ass_word(word: str) -> str:
    cleaned = re.sub(r"[\{\}\\\r\n]", "", word).strip().upper()
    return cleaned if cleaned else "..."

def escape_html(text: str) -> str:
    if not text:
        return ""
    return html.escape(str(text), quote=False)

def persist_status_msg_id(status_file: str, msg_id: str) -> None:
    try:
        os.makedirs(os.path.dirname(status_file) or ".", exist_ok=True)
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"status_msg_id": str(msg_id)}, f)
    except Exception as e:
        logger.warning(f"No se pudo persistir status_msg_id: {e}")

def load_status_msg_id(status_file: str) -> str:
    try:
        if os.path.exists(status_file):
            with open(status_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return str(data.get("status_msg_id", ""))
    except Exception:
        pass
    return ""

async def retry_async(
    fn: Callable[..., Any],
    *args: Any,
    retries: int = 3,
    backoff: float = 1.8,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    **kwargs: Any
) -> Any:
    delay = 1.0
    last_exception = None

    for attempt in range(1, retries + 1):
        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            else:
                return await asyncio.to_thread(fn, *args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt == retries:
                logger.error(f"Fallo definitivo en {fn.__name__} tras {retries} intentos: {e}")
                raise e
            logger.warning(f"Intento {attempt}/{retries} falló en {fn.__name__}: {e}. Reintentando en {delay:.1f}s...")
            await asyncio.sleep(delay)
            delay *= backoff

    if last_exception:
        raise last_exception

async def send_telegram(settings: Settings, text: str) -> None:
    logger.info(f"[Telegram Notify] {text[:120]}...")
    if not settings.bot_token or not settings.chat_id:
        return
    url = f"https://api.telegram.org/bot{settings.bot_token}/sendMessage"
    async with httpx.AsyncClient(timeout=12.0) as client:
        try:
            await client.post(url, json={
                "chat_id": int(settings.chat_id),
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": False
            })
        except Exception as e:
            logger.warning(f"No se pudo enviar notificación a Telegram: {e}")

async def update_live_progress(settings: Settings, percent: int, stage: str, detail: str = "") -> None:
    logger.info(f"[{percent}%] {stage} - {detail}")
    if not settings.bot_token or not settings.chat_id:
        return

    status_file = "output/.status_msg_id.json"
    current_msg_id = settings.status_msg_id or load_status_msg_id(status_file)

    filled = max(0, min(10, round(percent / 10)))
    bar = "🟩" * filled + "⬜" * (10 - filled)
    active_nets = [k.upper() for k, v in settings.platforms.items() if v]
    nets_str = " + ".join(active_nets) if active_nets else "YOUTUBE"

    safe_ch = escape_html(settings.channel_name)
    safe_topic = escape_html(settings.raw_topic)
    safe_stage = escape_html(stage)
    safe_detail = escape_html(detail)

    text = (
        f"🎬 <b>Fábrica Omnicanal — {safe_ch}</b>\n"
        f"{bar} <b>{percent}%</b>\n\n"
        f"📌 <b>Tema:</b> {safe_topic}\n"
        f"📦 <b>Cuota:</b> {settings.long_count} Largos + {settings.short_count} Shorts | 🕒 <b>Hora Pub:</b> {settings.publish_hour_art}\n"
        f"📱 <b>Redes:</b> {nets_str}\n"
        f"⚡ <b>Estado actual:</b> {safe_stage}\n"
        f"<i>{safe_detail}</i>"
    )

    async with httpx.AsyncClient(timeout=12.0) as client:
        if current_msg_id:
            try:
                url = f"https://api.telegram.org/bot{settings.bot_token}/editMessageText"
                r = await client.post(url, json={
                    "chat_id": int(settings.chat_id),
                    "message_id": int(current_msg_id),
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True
                })
                if r.status_code == 200:
                    return
            except Exception as e:
                logger.warning(f"No se pudo editar mensaje {current_msg_id}: {e}")

        try:
            url = f"https://api.telegram.org/bot{settings.bot_token}/sendMessage"
            r = await client.post(url, json={
                "chat_id": int(settings.chat_id),
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            })
            if r.status_code == 200:
                data = r.json()
                if data.get("ok") and data.get("result", {}).get("message_id"):
                    new_id = str(data["result"]["message_id"])
                    settings.status_msg_id = new_id
                    persist_status_msg_id(status_file, new_id)
        except Exception as e:
            logger.warning(f"No se pudo inicializar el panel de progreso: {e}")
