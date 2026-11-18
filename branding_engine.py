import os
import glob
import json
import urllib.parse
import asyncio
import logging
import httpx

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from config import Settings
from llm import ask_llm
from utils import send_telegram, retry_async, BROWSER_HEADERS, escape_html

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
    """Reconstruye el token OAuth desde BUNDLE_JSON, sube el Watermark y actualiza la pestaña 'Acerca de'."""
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
            logger.info(f"Token OAuth reconstruido para Branding en {token_path}")
        except Exception as e:
            logger.warning(f"No se pudo reconstruir el token desde BUNDLE_JSON: {e}")

    if not os.path.exists(token_path):
        all_tokens = sorted(glob.glob("tokens/*.json"))
        if all_tokens:
            token_path = all_tokens[0]
        else:
            return "⚠️ No se encontraron credenciales OAuth para aplicar los cambios en YouTube."

    def _execute_updates():
        creds = Credentials.from_authorized_user_file(
            token_path,
            [
                "https://www.googleapis.com/auth/youtube.force-ssl",
                "https://www.googleapis.com/auth/youtube.upload"
            ]
        )
        if not creds.valid or not creds.token:
            creds.refresh(Request())

        youtube = build("youtube", "v3", credentials=creds)
        results = []

        # 1. Aplicar Logotipo como Marca de Agua (Watermark)
        if os.path.exists(logo_path):
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
                results.append("✅ Logotipo aplicado como Marca de Agua oficial en los videos.")
            except Exception as e:
                logger.error(f"Error al aplicar Watermark: {e}")
                results.append(f"⚠️ Marca de agua pendiente (Verifica permiso youtube.force-ssl en OAuth).")

        # 2. Aplicar Biografía SEO (Pestaña 'Acerca de')
        if about_text:
            try:
                ch_res = youtube.channels().list(mine=True, part="snippet").execute()
                if ch_res.get("items"):
                    ch_item = ch_res["items"][0]
                    snippet = ch_item["snippet"]
                    snippet["description"] = about_text[:990]

                    youtube.channels().update(
                        part="snippet",
                        body={
                            "id": ch_item["id"],
                            "snippet": snippet
                        }
                    ).execute()
                    results.append("✅ Pestaña 'Acerca de' actualizada automáticamente en YouTube.")
            except Exception as e:
                logger.error(f"Error al actualizar Biografía (Snippet): {e}")
                results.append("⚠️ Biografía lista en el reporte (requiere re-autorizar OAuth con force-ssl para auto-escritura).")

        return "\n".join(results)

    try:
        import concurrent.futures
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, _execute_updates)
    except Exception as e:
        logger.error(f"Fallo general en APIs de YouTube: {e}")
        return "⚠️ Error al conectar con YouTube API."

async def build_channel_branding(settings: Settings) -> None:
    logger.info(f"Iniciando Motor de Branding para: {settings.channel_name} ({settings.raw_topic})")

    await send_telegram(
        settings,
        f"🎨 <b>Diseñando Identidad Omnicanal para {escape_html(settings.channel_name)}...</b>\n"
        f"🎯 <b>Nicho:</b> <i>{escape_html(settings.raw_topic)}</i>\n\n"
        f"<i>Generando estrategia SEO, paleta de colores, ganchos de retención y activos visuales (Logo y Banner)...</i>"
    )

    prompt_estrategia = (
        f"Actúa como un director creativo y experto en SEO para redes sociales. "
        f"Crea la identidad visual y estrategia para el canal '{settings.channel_name}' centrado en: '{settings.raw_topic}'.\n\n"
        f"Devuelve el resultado estrictamente en este formato:\n"
        f"PALETA: [3 Códigos HEX con su significado psicológico]\n"
        f"YOUTUBE_ABOUT: [Bio SEO persuasiva máximo 500 caracteres con palabras clave de alto tráfico para la pestaña Acerca de]\n"
        f"TIKTOK_BIO: [Bio punchy con CTA máximo 80 caracteres]\n"
        f"IG_BIO: [Bio atractiva en 3 líneas con emojis y CTA máximo 150 caracteres]\n"
        f"X_BIO: [Bio aguda y magnética máximo 160 caracteres]\n"
        f"GANCHOS: [3 fórmulas de ganchos virales de alta retención para los primeros 3 segundos]\n"
        f"KEYWORDS: [10 palabras clave separadas por comas]\n"
        f"LOGO_PROMPT: [Prompt en inglés, detallado, minimalista, fondo oscuro, vector art, sin texto]\n"
        f"BANNER_PROMPT: [Prompt en inglés, cinematográfico 16:9, alta resolución, sin texto]"
    )

    raw_strategy = await ask_llm(settings, prompt_estrategia, max_tokens=1100)

    logo_prompt = f"modern minimalist vector logo icon for {settings.raw_topic} brand, dark background, high contrast, 8k"
    banner_prompt = f"cinematic widescreen abstract atmosphere representing {settings.raw_topic}, dark moody lighting, 8k"
    youtube_about = ""

    parsing_about = False
    for line in raw_strategy.split('\n'):
        line_stripped = line.strip().replace('**', '')

        if line_stripped.startswith("LOGO_PROMPT:"):
            logo_prompt = line_stripped.replace("LOGO_PROMPT:", "").strip()
            parsing_about = False
        elif line_stripped.startswith("BANNER_PROMPT:"):
            banner_prompt = line_stripped.replace("BANNER_PROMPT:", "").strip()
            parsing_about = False
        elif line_stripped.startswith("YOUTUBE_ABOUT:"):
            youtube_about = line_stripped.replace("YOUTUBE_ABOUT:", "").strip()
            parsing_about = True
        elif parsing_about and line_stripped.startswith(("TIKTOK_BIO:", "IG_BIO:", "X_BIO:", "GANCHOS:", "KEYWORDS:", "PALETA:", "LOGO_PROMPT:", "BANNER_PROMPT:")):
            parsing_about = False
        elif parsing_about:
            youtube_about += "\n" + line_stripped

    youtube_about = youtube_about.strip()

    os.makedirs("output/branding", exist_ok=True)
    logo_path = f"output/branding/{settings.channel_id}_logo.jpg"
    banner_path = f"output/branding/{settings.channel_id}_banner.jpg"

    logger.info("Renderizando Logo (512x512) y Banner (1920x1080)...")
    await asyncio.gather(
        generate_visual_asset(logo_prompt, logo_path, 512, 512),
        generate_visual_asset(banner_prompt, banner_path, 1920, 1080)
    )

    await send_telegram(settings, f"⚙️ <i>Aplicando marca de agua y biografía SEO en el perfil de YouTube de {escape_html(settings.channel_name)}...</i>")
    api_result = await apply_branding_and_bio_to_youtube(settings, logo_path, youtube_about)

    report = (
        f"🚀 <b>BRAND KIT OMNICANAL: {escape_html(settings.channel_name)}</b>\n\n"
        f"{escape_html(api_result)}\n\n"
        f"<b>Estrategia y Copywriting Multi-Plataforma:</b>\n"
        f"<pre>{escape_html(raw_strategy[:2400])}</pre>\n\n"
        f"<i>📦 El Logo (512x512) y el Banner (1920x1080) están disponibles en los Artifacts de esta ejecución en GitHub Actions.</i>"
    )

    await send_telegram(settings, report)
    logger.info("Branding Kit generado, aplicado y notificado con éxito.")
