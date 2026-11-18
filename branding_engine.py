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
    """Genera activos visuales (Logos, Banners) utilizando Pollinations AI."""
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
    
    await send_telegram(
        settings, 
        f"🎨 <b>Diseñando Identidad para {settings.channel_name}...</b>\n"
        f"<i>Generando estrategia SEO omnicanal, paleta de colores y activos visuales (Logo y Banner). Esto tomará unos 30 segundos.</i>"
    )

    # 1. Estrategia SEO y Copywriting Omnicanal
    prompt_estrategia = (
        f"Actúa como un director creativo y experto en SEO para redes sociales. "
        f"Crea la identidad visual y estrategia de posicionamiento para un canal de contenido "
        f"llamado '{settings.channel_name}' centrado en el nicho: '{settings.raw_topic}'.\n\n"
        f"Devuelve el resultado estrictamente en este formato (usa los prefijos exactos):\n"
        f"PALETA: [Sugiere 3 Códigos HEX con una breve justificación de su impacto psicológico en este nicho]\n"
        f"YOUTUBE_ABOUT: [Descripción SEO persuasiva para la pestaña 'Acerca de' en YouTube, máximo 500 caracteres, incluyendo palabras clave del nicho]\n"
        f"TIKTOK_BIO: [Bio punchy con Call to Action, máximo 80 caracteres]\n"
        f"X_BIO: [Bio aguda, profesional y que genere intriga, máximo 160 caracteres]\n"
        f"GANCHOS: [Proporciona 3 estructuras o fórmulas de ganchos virales específicos para retener la atención en este nicho]\n"
        f"LOGO_PROMPT: [Prompt en inglés, detallado, estilo minimalista y moderno, fondo sólido oscuro, para generar el logo usando una IA de imágenes]\n"
        f"BANNER_PROMPT: [Prompt en inglés, estilo cinematográfico o abstracto, apaisado, que capture la esencia del canal, sin texto, para el banner del canal]"
    )
    
    raw_strategy = await ask_llm(settings, prompt_estrategia, max_tokens=1000)
    
    # Valores por defecto en caso de que el LLM no devuelva la estructura exacta
    logo_prompt = f"modern minimalist logo for a {settings.raw_topic} channel, dark background, vector art, high resolution"
    banner_prompt = f"cinematic abstract background representing {settings.raw_topic}, dark theme, high contrast, 8k resolution"
    
    # Extraer los prompts visuales de la respuesta del LLM
    for line in raw_strategy.split('\n'):
        line_stripped = line.strip()
        if line_stripped.startswith("LOGO_PROMPT:"):
            logo_prompt = line_stripped.replace("LOGO_PROMPT:", "").strip()
        elif line_stripped.startswith("BANNER_PROMPT:"):
            banner_prompt = line_stripped.replace("BANNER_PROMPT:", "").strip()

    # 2. Generación Concurrente de Activos Visuales
    os.makedirs("output/branding", exist_ok=True)
    logo_path = f"output/branding/{settings.channel_id}_logo.jpg"
    banner_path = f"output/branding/{settings.channel_id}_banner.jpg"
    
    logger.info("Generando Logo y Banner...")
    await asyncio.gather(
        generate_visual_asset(logo_prompt, logo_path, 512, 512),
        generate_visual_asset(banner_prompt, banner_path, 1920, 1080)
    )

    # 3. Empaquetado y Notificación
    report = (
        f"🚀 <b>BRAND KIT OMNICANAL: {settings.channel_name}</b>\n\n"
        f"<b>Estrategia y Copywriting:</b>\n"
        f"<pre>{raw_strategy[:2500]}</pre>\n\n"
        f"<i>Nota: El Logotipo (512x512) y el Banner (1920x1080) se han renderizado y guardado en la carpeta 'output/branding/' de la máquina virtual. "
        f"Puedes revisar los artefactos del workflow de GitHub Actions para descargarlos (si tienes configurada la subida de artefactos).</i>"
    )
    
    await send_telegram(settings, report)
    logger.info("Branding Kit generado y notificado con éxito a Telegram.")
