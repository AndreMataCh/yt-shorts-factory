import os
import urllib.parse
import asyncio
import logging
import httpx
from config import Settings
from llm import ask_llm
from utils import send_telegram, retry_async, BROWSER_HEADERS

logger = logging.getLogger("PAULA.Branding")

async def generate_visual_asset(prompt: str, filename: str, w: int, h: int) -> str:
    """Genera logos y banners usando inteligencia artificial visual."""
    safe_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width={w}&height={h}&nologo=true"
    
    async def _fetch():
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
            res = await client.get(url)
            res.raise_for_status()
            with open(filename, "wb") as f:
                f.write(res.content)
            return filename

    try:
        return await retry_async(_fetch, retries=3, backoff=2.0)
    except Exception as e:
        logger.error(f"Fallo al generar asset {filename}: {e}")
        return ""

async def build_channel_branding(settings: Settings) -> None:
    logger.info(f"Iniciando Motor de Branding para: {settings.channel_name} ({settings.raw_topic})")
    
    # 1. Estrategia SEO y Copywriting Omnicanal
    prompt_estrategia = (
        f"Actúa como un experto en Branding y SEO para redes sociales. Crea la identidad visual y "
        f"estrategia de posicionamiento para un nuevo canal llamado '{settings.channel_name}' centrado en '{settings.raw_topic}'.\n\n"
        f"Devuelve el resultado estrictamente en este formato:\n"
        f"PALETA: [3 Códigos HEX con su significado psicológico]\n"
        f"YOUTUBE_ABOUT: [Descripción SEO para la pestaña 'Acerca de', máx 500 caracteres con palabras clave]\n"
        f"TIKTOK_BIO: [Bio punchy con CTA, máx 80 caracteres]\n"
        f"X_BIO: [Bio aguda y profesional, máx 160 caracteres]\n"
        f"GANCHOS: [3 estructuras de ganchos virales específicos para este nicho]\n"
        f"LOGO_PROMPT: [Prompt en inglés, detallado, estilo minimalista/moderno, fondo sólido para generar el logo]\n"
        f"BANNER_PROMPT: [Prompt en inglés, cinematográfico, paisaje horizontal, sin texto]"
    )
    
    raw_strategy = await ask_llm(settings, prompt_estrategia, max_tokens=1000)
    
    # Extraer prompts visuales
    logo_prompt = "modern minimalist logo for a tech channel, flat vector"
    banner_prompt = "cinematic abstract landscape, dark background, neon accents, 8k"
    
    for line in raw_strategy.split('\n'):
        if line.startswith("LOGO_PROMPT:"):
            logo_prompt = line.replace("LOGO_PROMPT:", "").strip()
        elif line.startswith("BANNER_PROMPT:"):
            banner_prompt = line.replace("BANNER_PROMPT:", "").strip()

    await send_telegram(settings, f"🎨 <b>Diseñando Identidad para {settings.channel_name}...</b>\n<i>Generando estrategia SEO, paleta de colores y activos visuales (Logo y Banner). Esto tomará unos 30 segundos.</i>")

    # 2. Generación Concurrente de Activos Visuales
    os.makedirs("output/branding", exist_ok=True)
    logo_path = f"output/branding/{settings.channel_id}_logo.jpg"
    banner_path = f"output/branding/{settings.channel_id}_banner.jpg"
    
    await asyncio.gather(
        generate_visual_asset(logo_prompt, logo_path, 512, 512),
        generate_visual_asset(banner_prompt, banner_path, 1920, 1080)
    )

    # 3. Empaquetado y Notificación
    report = (
        f"🚀 <b>BRAND KIT OMNICANAL: {settings.channel_name}</b>\n\n"
        f"<b>Estrategia Extraída:</b>\n"
        f"<pre>{raw_strategy[:2500]}</pre>\n\n"
        f"<i>Nota: Logo y Banner guardados en la carpeta output/branding/. Revisa los logs para descargar los archivos si estás en GitHub Actions.</i>"
    )
    
    await send_telegram(settings, report)
    logger.info("Branding Kit generado y notificado con éxito.")
