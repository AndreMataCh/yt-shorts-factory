import os
import glob
import urllib.parse
import asyncio
import logging
import httpx
import re

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

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

async def apply_branding_and_bio_to_youtube(settings: Settings, logo_path: str, about_text: str) -> str:
    """Sube el logotipo (Watermark) y actualiza la pestaña 'Acerca de' (Snippet Description)."""
    token_path = f"tokens/{settings.channel_id}_token_kv.json"
    
    if not os.path.exists(token_path):
        all_tokens = sorted(glob.glob("tokens/*.json"))
        if all_tokens:
            token_path = all_tokens[0]
        else:
            return "⚠️ No se encontraron credenciales (token OAuth) para aplicar los cambios."

    def _execute_updates():
        # Solicitamos permisos de force-ssl, requeridos para la API de Branding y edición de canales
        creds = Credentials.from_authorized_user_file(
            token_path, 
            ["https://www.googleapis.com/auth/youtube.force-ssl"]
        )
        if not creds.valid or not creds.token:
            creds.refresh(Request())

        youtube = build("youtube", "v3", credentials=creds)
        results = []
        
        # 1. Aplicar Logotipo (Marca de Agua)
        try:
            media = MediaFileUpload(logo_path, chunksize=-1, resumable=True, mimetype="image/jpeg")
            youtube.watermarks().set(
                channelId=settings.channel_id,
                body={
                    "timing": {
                        "type": "offsetFromStart",
                        "offsetMs": 1000,
                        "durationMs": 10000 
                    },
                    "position": {
                        "type": "corner",
                        "cornerPosition": "bottomRight"
                    }
                },
                media_body=media
            ).execute()
            results.append("✅ Logotipo aplicado como Marca de Agua en los videos.")
        except Exception as e:
            logger.error(f"Error al aplicar Watermark: {e}")
            results.append("⚠️ Error al aplicar la marca de agua (posible falta de permisos).")

        # 2. Aplicar Biografía SEO (Pestaña 'Acerca de')
        if about_text:
            try:
                # Descargamos el snippet actual del canal para modificar SOLO la descripción sin pisar el título
                ch_res = youtube.channels().list(mine=True, part="snippet").execute()
                if ch_res.get("items"):
                    ch_item = ch_res["items"][0]
                    snippet = ch_item["snippet"]
                    snippet["description"] = about_text[:990]  # Respetando límites de la API
                    
                    # Realizamos el PUT con el snippet modificado
                    youtube.channels().update(
                        part="snippet",
                        body={
                            "id": ch_item["id"],
                            "snippet": snippet
                        }
                    ).execute()
                    results.append("✅ Pestaña 'Acerca de' actualizada con la nueva biografía SEO.")
            except Exception as e:
                logger.error(f"Error al aplicar Biografía (Snippet): {e}")
                results.append("⚠️ Error al actualizar la biografía (verifica permisos de lectura/escritura).")

        return "\n".join(results)

    try:
        import concurrent.futures
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, _execute_updates)
            return result
    except Exception as e:
        logger.error(f"Fallo general en APIs de YouTube: {e}")
        return "⚠️ Error crítico al conectar con YouTube APIs."


async def build_channel_branding(settings: Settings) -> None:
    logger.info(f"Iniciando Motor de Branding para: {settings.channel_name} ({settings.raw_topic})")
    
    await send_telegram(
        settings, 
        f"🎨 <b>Diseñando Identidad para {settings.channel_name}...</b>\n"
        f"<i>Generando estrategia SEO omnicanal, paleta de colores y activos visuales (Logo y Banner). Esto tomará unos 30 segundos.</i>"
    )

    # 1. Generación de Estrategia SEO (LLM)
    prompt_estrategia = (
        f"Actúa como un director creativo y experto en SEO para redes sociales. "
        f"Crea la identidad visual y estrategia para un canal llamado '{settings.channel_name}' centrado en: '{settings.raw_topic}'.\n\n"
        f"Devuelve el resultado estrictamente en este formato:\n"
        f"PALETA: [3 Códigos HEX con su significado]\n"
        f"YOUTUBE_ABOUT: [Bio SEO máximo 500 caracteres, con palabras clave para la pestaña Acerca de]\n"
        f"TIKTOK_BIO: [Bio punchy máximo 80 caracteres]\n"
        f"X_BIO: [Bio aguda máximo 160 caracteres]\n"
        f"GANCHOS: [3 fórmulas de ganchos virales]\n"
        f"LOGO_PROMPT: [Prompt detallado, fondo oscuro, vector]\n"
        f"BANNER_PROMPT: [Prompt cinematográfico, paisaje, sin texto]"
    )
    
    raw_strategy = await ask_llm(settings, prompt_estrategia, max_tokens=1000)
    
    # Valores de resguardo
    logo_prompt = f"modern minimalist logo for a {settings.raw_topic} channel, dark background, vector art"
    banner_prompt = f"cinematic abstract background representing {settings.raw_topic}, dark theme, 8k"
    youtube_about = ""
    
    # Extractor Robusto (Soporta múltiples líneas si el LLM incluye saltos de línea en la biografía)
    parsing_about = False
    for line in raw_strategy.split('\n'):
        line_stripped = line.strip().replace('**', '')
        
        if line_stripped.startswith("LOGO_PROMPT:"):
            logo_prompt = line_stripped.replace("LOGO_PROMPT:", "").strip()
        elif line_stripped.startswith("BANNER_PROMPT:"):
            banner_prompt = line_stripped.replace("BANNER_PROMPT:", "").strip()
        elif line_stripped.startswith("YOUTUBE_ABOUT:"):
            youtube_about = line_stripped.replace("YOUTUBE_ABOUT:", "").strip()
            parsing_about = True
        elif parsing_about and line_stripped.startswith(("TIKTOK_BIO:", "X_BIO:", "GANCHOS:", "PALETA:", "LOGO_PROMPT:")):
            parsing_about = False
        elif parsing_about:
            youtube_about += "\n" + line_stripped

    youtube_about = youtube_about.strip()

    # 2. Generación Visual Concurrente
    os.makedirs("output/branding", exist_ok=True)
    logo_path = f"output/branding/{settings.channel_id}_logo.jpg"
    banner_path = f"output/branding/{settings.channel_id}_banner.jpg"
    
    logger.info("Generando Logo y Banner...")
    await asyncio.gather(
        generate_visual_asset(logo_prompt, logo_path, 512, 512),
        generate_visual_asset(banner_prompt, banner_path, 1920, 1080)
    )

    # 3. Aplicación Automática en YouTube
    await send_telegram(settings, "⚙️ <i>Aplicando diseño de marca y metadatos SEO en el perfil de YouTube...</i>")
    api_result = await apply_branding_and_bio_to_youtube(settings, logo_path, youtube_about)

    # 4. Notificación Final
    report = (
        f"🚀 <b>BRAND KIT OMNICANAL: {settings.channel_name}</b>\n\n"
        f"{api_result}\n\n"
        f"<b>Estrategia y Copywriting:</b>\n"
        f"<pre>{raw_strategy[:2500]}</pre>\n\n"
        f"<i>Nota: El Logo original y el Banner están listos para descarga en Artifacts de GitHub Actions.</i>"
    )
    
    await send_telegram(settings, report)
    logger.info("Branding Kit generado, aplicado y notificado con éxito.")
