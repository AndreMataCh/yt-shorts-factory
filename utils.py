import os
import re
import html
import json
import asyncio
import logging
import time
from typing import Callable, Any, Type

logger = logging.getLogger("PAULA.Utils")

def format_ass_time(seconds: float) -> str:
    """Convierte marcas de tiempo en formato estándar de subtítulos ASS."""
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
    """Escapa caracteres especiales para evitar inyecciones o fallos en parse_mode HTML de Telegram."""
    if not text:
        return ""
    return html.escape(str(text), quote=False)

def persist_status_msg_id(status_file: str, msg_id: str) -> None:
    """Guarda el ID del mensaje de progreso en disco para recuperación ante reinicios del job."""
    try:
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"status_msg_id": str(msg_id)}, f)
    except Exception as e:
        logger.warning(f"No se pudo persistir status_msg_id: {e}")

def load_status_msg_id(status_file: str) -> str:
    """Carga el ID del mensaje de progreso previo si existe."""
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
    """
    Ejecuta una función asíncrona con reintentos controlados (backoff exponencial).
    Ideal para APIs externas frágiles (Gemini, Groq, Pexels, YouTube, Telegram).
    """
    delay = 1.0
    last_exception = None

    for attempt in range(1, retries + 1):
        try:
            if asyncio.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            else:
                return fn(*args, **kwargs)
        except exceptions as e:
            last_exception = e
            if attempt == retries:
                logger.error(f"Fallo definitivo en {fn.__name__} tras {retries} intentos. Error: {e}")
                raise e
            logger.warning(f"Intento {attempt}/{retries} falló en {fn.__name__}: {e}. Reintentando en {delay:.1f}s...")
            await asyncio.sleep(delay)
            delay *= backoff

    if last_exception:
        raise last_exception
# utils.py (Añadir al final del archivo)

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}

API_HEADERS = {
    "User-Agent": "PAULA-Enterprise-Engine/8.0",
    "Accept": "application/json",
    "Content-Type": "application/json"
}
