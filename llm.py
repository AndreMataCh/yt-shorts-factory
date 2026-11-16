import os
import re
import logging
import httpx
from typing import Tuple, List
from config import Settings
from utils import retry_async, API_HEADERS

logger = logging.getLogger("PAULA.LLM")

def clean_llm_output(text: str) -> str:
    """Elimina etiquetas de pensamiento (<think>) y bloques residuales de los LLMs."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()

def parse_script_and_sfx(raw_script: str) -> Tuple[str, List[dict]]:
    """
    CORRECCIÓN SFX: Parsea el guion extrayendo los marcadores [SFX: ...] para el mezclador de audio,
    y devuelve el texto LIMPIO para que el motor de voz (TTS) NUNCA los lea en voz alta.
    """
    pattern = re.compile(r"\[SFX:\s*(whoosh|impact|riser|glitch)\]", flags=re.IGNORECASE)
    tokens = raw_script.split()
    clean_words = []
    sfx_cues = []

    for token in tokens:
        match = pattern.search(token)
        if match:
            sfx_type = match.group(1).lower()
            sfx_cues.append({"type": sfx_type, "word_index": len(clean_words)})
            rem = pattern.sub("", token).strip()
            if rem:
                clean_words.append(rem)
        else:
            clean_t = re.sub(r"[\*\#\[\]\(\)]", "", token).strip()
            if clean_t:
                clean_words.append(clean_t)

    if not sfx_cues and len(clean_words) > 10:
        sfx_cues.append({"type": "impact", "word_index": 0})
        sfx_cues.append({"type": "whoosh", "word_index": len(clean_words) // 3})
        sfx_cues.append({"type": "riser", "word_index": (len(clean_words) * 2) // 3})

    return " ".join(clean_words), sfx_cues

async def ask_llm(settings: Settings, prompt: str, max_tokens: int = 1200) -> str:
    """Orquestador LLM multi-proveedor con reintentos y autodescubrimiento."""
    errors = []

    async def _call_gemini() -> str:
        if not settings.gemini_api_key or not settings.gemini_api_key.startswith("AIza"):
            raise ValueError("Gemini key no configurada")
        headers = {**API_HEADERS, "x-goog-api-key": settings.gemini_api_key}
        models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            for model in models:
                res = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    json={"contents": [{"parts": [{"text": prompt}]}]}
                )
                if res.status_code == 200:
                    candidates = res.json().get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        cleaned = clean_llm_output(content)
                        if cleaned:
                            return cleaned
        raise RuntimeError("Fallo en todos los modelos Gemini")

    async def _call_groq() -> str:
        if not settings.groq_api_key or not settings.groq_api_key.startswith("gsk_"):
            raise ValueError("Groq key no configurada")
        headers = {**API_HEADERS, "Authorization": f"Bearer {settings.groq_api_key}"}
        models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]
        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            for model in models:
                res = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": max_tokens}
                )
                if res.status_code == 200:
                    content = res.json()["choices"][0]["message"].get("content") or ""
                    cleaned = clean_llm_output(content)
                    if cleaned:
                        return cleaned
        raise RuntimeError("Fallo en Groq")

    try:
        return await retry_async(_call_gemini, retries=2)
    except Exception as e:
        errors.append(str(e))

    try:
        return await retry_async(_call_groq, retries=2)
    except Exception as e:
        errors.append(str(e))

    logger.warning(f"APIs remotas fallaron ({errors}). Usando fallback determinista local.")
    return _build_fallback(prompt, settings.raw_topic, settings.channel_name)

def _build_fallback(prompt: str, topic: str, channel_name: str) -> str:
    p_low = prompt.lower()
    if "comma-separated" in p_low:
        return (
            "cinematic dark room dramatic lighting, high tech holographic interface closeup, "
            "mysterious figure analyzing data screens, futuristic neural network glowing nodes, "
            "aerial view modern metropolis night, intense interrogation spotlight shadow, "
            "digital security breach red alert, ancient secret manuscript macro shot"
        )
    if "título" in p_low or "titulo" in p_low:
        return f"La Verdad Oculta sobre {topic[:42]}"
    return (
        f"[SFX: impact] Lo que estás a punto de descubrir sobre {topic} cambiará por completo tu perspectiva. "
        f"[SFX: whoosh] Durante años, los especialistas de {channel_name} analizaron este fenómeno sin hallar respuestas sencillas. "
        f"[SFX: riser] Sin embargo, las pruebas ocultas revelan un patrón fascinante que opera detrás de escena. "
        f"[SFX: impact] Descubre el análisis completo en el enlace de la descripción."
    )
