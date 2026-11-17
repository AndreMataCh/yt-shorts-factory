import os
import re
import logging
import httpx
from typing import Tuple, List
from config import Settings
from utils import retry_async, API_HEADERS

logger = logging.getLogger("PAULA.LLM")

def clean_llm_output(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()

def get_seo_context() -> str:
    if os.path.exists("seo_context.md"):
        try:
            with open("seo_context.md", "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return f"\n\nContexto competitivo extraído en tiempo real:\n{content[:1500]}\n"
        except Exception:
            pass
    return ""

def parse_script_and_sfx(raw_script: str) -> Tuple[str, List[dict]]:
    """
    Extrae todas las etiquetas [SFX: ...] o variantes sueltas para que el sintetizador de voz
    NUNCA las lea como texto narrativo, y devuelve sus posiciones exactas para FFmpeg.
    """
    # Normalizar formatos como [SFX: impact] o (SFX: whoosh)
    normalized = re.sub(
        r"[\[\(]\s*SFX\s*:\s*(whoosh|impact|riser|glitch)\s*[\]\)]",
        r" [SFX:\1] ",
        raw_script,
        flags=re.IGNORECASE
    )
    # Eliminar cualquier otro marcador entre corchetes que el LLM haya inventado (ej. [Música], [Pausa])
    normalized = re.sub(r"\[(?!SFX:)(.*?)\]", " ", normalized)

    pattern = re.compile(r"\[SFX:(whoosh|impact|riser|glitch)\]", flags=re.IGNORECASE)
    tokens = normalized.split()
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
            if clean_t and not clean_t.lower().startswith("sfx:"):
                clean_words.append(clean_t)

    if not sfx_cues and len(clean_words) > 10:
        sfx_cues.append({"type": "impact", "word_index": 0})
        sfx_cues.append({"type": "whoosh", "word_index": len(clean_words) // 3})
        sfx_cues.append({"type": "riser", "word_index": (len(clean_words) * 2) // 3})

    return " ".join(clean_words), sfx_cues

async def ask_llm(settings: Settings, prompt: str, max_tokens: int = 1200) -> str:
    errors = []

    if settings.gemini_api_key and settings.gemini_api_key.startswith("AIza"):
        gemini_headers = {**API_HEADERS, "x-goog-api-key": settings.gemini_api_key}
        gemini_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        async with httpx.AsyncClient(timeout=30.0, headers=gemini_headers) as client:
            for model in gemini_models:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                    res = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
                    if res.status_code == 200:
                        candidates = res.json().get("candidates", [])
                        if candidates:
                            content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                            cleaned = clean_llm_output(content)
                            if cleaned:
                                logger.info(f"Respuesta exitosa desde Gemini ({model})")
                                return cleaned
                    else:
                        errors.append(f"Gemini ({model}): HTTP {res.status_code}")
                except Exception as e:
                    errors.append(f"Gemini ({model}): {str(e)[:40]}")

    if settings.groq_api_key and settings.groq_api_key.startswith("gsk_"):
        groq_headers = {**API_HEADERS, "Authorization": f"Bearer {settings.groq_api_key}"}
        groq_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "qwen/qwen3-32b",
            "gemma2-9b-it"
        ]
        async with httpx.AsyncClient(timeout=30.0, headers=groq_headers) as client:
            try:
                m_res = await client.get("https://api.groq.com/openai/v1/models", timeout=8.0)
                if m_res.status_code == 200:
                    live_models = [
                        m.get("id", "")
                        for m in m_res.json().get("data", [])
                        if m.get("id") and not any(x in m.get("id", "").lower() for x in ["whisper", "guard", "tts", "embed", "vision", "audio", "playai"])
                    ]
                    if live_models:
                        ordered = [m for m in groq_models if m in live_models]
                        for lm in live_models:
                            if lm not in ordered:
                                ordered.append(lm)
                        groq_models = ordered
            except Exception:
                pass

            for model in groq_models[:5]:
                try:
                    res = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.7,
                            "max_tokens": max_tokens
                        }
                    )
                    if res.status_code == 200:
                        content = res.json()["choices"][0]["message"].get("content") or ""
                        cleaned = clean_llm_output(content)
                        if cleaned:
                            logger.info(f"Respuesta exitosa desde Groq ({model})")
                            return cleaned
                    else:
                        errors.append(f"Groq ({model}): HTTP {res.status_code}")
                except Exception as e:
                    errors.append(f"Groq ({model}): {str(e)[:40]}")

    if settings.openrouter_api_key:
        or_headers = {**API_HEADERS, "Authorization": f"Bearer {settings.openrouter_api_key}"}
        async with httpx.AsyncClient(timeout=30.0, headers=or_headers) as client:
            for or_model in ["openrouter/auto", "meta-llama/llama-3.3-70b-instruct:free"]:
                try:
                    res = await client.post("https://openrouter.ai/api/v1/chat/completions", json={
                        "model": or_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7
                    })
                    if res.status_code == 200:
                        content = res.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                        cleaned = clean_llm_output(content)
                        if cleaned:
                            logger.info(f"Respuesta exitosa desde OpenRouter ({or_model})")
                            return cleaned
                except Exception as e:
                    errors.append(f"OpenRouter: {str(e)[:40]}")

    async with httpx.AsyncClient(timeout=25.0, headers=API_HEADERS, follow_redirects=True) as client:
        for poly_url in ["https://text.pollinations.ai/openai", "https://text.pollinations.ai/"]:
            try:
                res = await client.post(
                    poly_url,
                    json={"model": "openai", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
                )
                if res.status_code == 200:
                    raw_txt = res.text.strip()
                    try:
                        data = res.json()
                        if "choices" in data and len(data["choices"]) > 0:
                            content = data["choices"][0].get("message", {}).get("content", "")
                            cleaned = clean_llm_output(content)
                            if cleaned:
                                return cleaned
                    except Exception:
                        cleaned = clean_llm_output(raw_txt)
                        if len(cleaned) > 15:
                            return cleaned
            except Exception as e:
                errors.append(f"Pollinations: {str(e)[:35]}")

    logger.warning(f"APIs remotas fallaron ({errors}). Usando fallback determinista local.")
    return _build_fallback(prompt, settings.raw_topic, settings.channel_name)

def _build_fallback(prompt: str, topic: str, channel_name: str) -> str:
    p_low = prompt.lower()
    if "comma-separated" in p_low:
        return (
            "cinematic dark room dramatic lighting, high tech holographic interface closeup, "
            "mysterious figure analyzing data screens, futuristic neural network glowing nodes, "
            "aerial view modern metropolis night, intense interrogation spotlight shadow, "
            "digital security breach red alert, ancient secret manuscript macro shot, "
            "hyperrealistic laboratory experiment, dramatic reveal camera zoom"
        )
    if "título" in p_low or "titulo" in p_low:
        return f"La Verdad Oculta sobre {topic[:42]}"
    if "punto y coma" in p_low:
        return f"{topic} Enfoque 1; {topic} Enfoque 2; {topic} Enfoque 3; {topic} Enfoque 4"
    return (
        f"[SFX: impact] Lo que estás a punto de descubrir sobre {topic} cambiará por completo tu perspectiva. "
        f"[SFX: whoosh] Durante años, los especialistas de {channel_name} analizaron cada detalle detrás de este fenómeno sin encontrar una explicación sencilla. "
        f"[SFX: riser] Sin embargo, cuando observamos las pruebas ocultas y los patrones que se repiten en los casos más recientes, emerge una revelación impactante. "
        f"[SFX: glitch] La verdadera clave no está en lo que parece evidente a simple vista, sino en el mecanismo silencioso que opera detrás de escena. "
        f"[SFX: impact] Descarga la guía oficial y descubre el análisis completo en el enlace de la descripción."
    )

async def generate_long_plan(settings: Settings, topic: str) -> dict:
    seo_intel = get_seo_context()
    script_prompt = (
        f"Eres el guionista principal del canal '{settings.channel_name}'. Escribe un guion documental hipnótico en español neutro de 320 a 400 palabras sobre: '{topic}'. "
        f"{seo_intel}"
        "ESTRUCTURA DE ALTA RETENCIÓN OBLIGATORIA:\n"
        "1. Segundos 0-5: Gancho de contradicción o pregunta imposible de ignorar (empieza con [SFX: impact]).\n"
        "2. Cada 60 palabras abre un nuevo bucle de misterio (Open Loop) antes de cerrar el anterior.\n"
        "3. Inserta entre 6 y 8 marcadores exactos [SFX: impact], [SFX: whoosh], [SFX: riser] o [SFX: glitch] en las revelaciones clave.\n"
        "4. En el último párrafo invita sutilmente a descargar el dossier/plantilla oficial en la descripción.\n"
        "REGLA ESTRICTA: Devuelve ÚNICAMENTE las palabras que dirá el narrador con sus etiquetas [SFX: ...]. Sin títulos, sin asteriscos, sin notas de cámara."
    )
    raw_script = await ask_llm(settings, script_prompt, max_tokens=1300)
    clean_script, sfx_cues = parse_script_and_sfx(raw_script)

    title = settings.custom_title or (
        await ask_llm(settings, f"Crea un título magnético de alto CTR en YouTube (máx 62 caracteres, con brecha de curiosidad) sobre '{topic}'. SOLO el título sin comillas.", max_tokens=150)
    ).replace('"', '').splitlines()[0].strip()[:75]

    terms_raw = await ask_llm(
        settings,
        f"Based on this documentary script about '{topic}': '{clean_script[:450]}', provide 12 chronological visual scene prompts in English (4 to 7 words each, hyperrealistic cinematic lighting, specific subject action). Return ONLY the 12 comma-separated prompts.",
        max_tokens=300
    )
    terms = [t.strip() for t in terms_raw.split(",") if t.strip()][:12]
    while len(terms) < 12:
        terms.append(f"cinematic dramatic scene {topic}")

    monetization_block = f"\n\n🔥 Descarga la Plantilla / Dossier Oficial de este episodio aquí:\n👉 {settings.monetization_url}" if settings.monetization_url else ""
    base_desc = settings.custom_description or f"{title}\n\nExploramos a fondo: {topic}.\n\n{clean_script[:250]}..."

    return {
        "title": title,
        "description": f"{base_desc}{monetization_block}\n\n#Documental #{settings.channel_name.replace(' ', '')} #Viral",
        "tags": ["documental", "misterio", "tecnologia", "viral", settings.channel_name.lower()],
        "script": clean_script,
        "sfx_cues": sfx_cues,
        "search_terms": terms
    }

async def generate_teaser_short_plan(settings: Settings, topic: str, long_title: str = "", long_url: str = "") -> dict:
    seo_intel = get_seo_context()
    if settings.monetization_url:
        cta_line = "Descarga la guía oficial y mira el análisis completo en el enlace del perfil."
    elif long_url:
        cta_line = "Mira el caso completo en el enlace fijado en la descripción."
    else:
        cta_line = "Síguenos ahora porque la segunda parte te dejará sin palabras."

    script_prompt = (
        f"Escribe un guion vertical ultra-adictivo de 30 segundos (75-90 palabras) optimizado para YouTube Shorts, TikTok e Instagram Reels del canal '{settings.channel_name}' sobre '{topic}'. "
        f"{seo_intel}"
        f"La primera frase debe ser un gancho explosivo con [SFX: impact]. Usa ritmo trepidante, inserta 4 marcadores [SFX: impact], [SFX: whoosh], [SFX: riser] o [SFX: glitch] y cierra con: '{cta_line}' "
        "REGLA ESTRICTA: Devuelve ÚNICAMENTE el texto hablado en español neutro con las etiquetas [SFX: ...], sin asteriscos ni emojis."
    )
    raw_script = await ask_llm(settings, script_prompt, max_tokens=500)
    clean_script, sfx_cues = parse_script_and_sfx(raw_script)

    title = (
        await ask_llm(settings, f"Crea un título corto y viral (máx 55 caracteres) para Shorts/TikTok/Reels sobre '{topic}'. SOLO el título sin comillas.", max_tokens=150)
    ).replace('"', '').splitlines()[0].strip()[:65]

    terms_raw = await ask_llm(
        settings,
        f"Based on this short story about '{topic}': '{clean_script}', provide 8 chronological visual scene descriptions in English (4 to 7 words each, intense cinematic atmosphere). Return ONLY the 8 comma-separated descriptions.",
        max_tokens=250
    )
    terms = [t.strip() for t in terms_raw.split(",") if t.strip()][:8]
    while len(terms) < 8:
        terms.append("cinematic suspenseful dark scene")

    x_post = (
        await ask_llm(settings, f"Escribe un post viral e intrigante para X (Twitter) de máx 240 caracteres sobre '{topic}' con 2 hashtags. SOLO el texto.", max_tokens=140)
    ).strip()[:260]

    desc_link = f"\n\n🔗 Mira el Documental Completo aquí: {long_url}" if long_url else ""
    monetization_block = f"\n📦 Plantilla / Dossier Oficial: {settings.monetization_url}" if settings.monetization_url else ""
    return {
        "title": title,
        "description": f"{title}{desc_link}{monetization_block}\n\n#Shorts #TikTok #Reels #{settings.channel_name.replace(' ', '')}",
        "tiktok_caption": f"{title} 🔥 {settings.monetization_url} #fyp #parati #viral #{settings.channel_name.replace(' ', '').lower()}",
        "reels_caption": f"{title}\n.\n{settings.monetization_url}\n#reels #viral #documental",
        "x_post": f"{x_post}\n{long_url or settings.monetization_url}".strip(),
        "tags": ["shorts", "tiktok", "reels", "viral", settings.channel_name.lower()],
        "script": clean_script,
        "sfx_cues": sfx_cues,
        "search_terms": terms
    }
