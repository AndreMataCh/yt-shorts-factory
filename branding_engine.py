import os
import re
import glob
import json
import random
import subprocess
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


def build_niche_visual_presets(channel_name: str, topic: str) -> tuple[str, str]:
    """Construye prompts visuales en inglés de alta precisión según la identidad del canal."""
    sig = f"{channel_name} {topic}".lower()
    if any(k in sig for k in ["interrogatorio", "crimen", "forense", "policia", "misterio", "detective"]):
        logo_p = (
            "minimalist iconic Esport & Netflix true crime emblem logo, dramatic detective silhouette "
            "and interrogation lamp beam, obsidian black background, crimson red and cold steel rim light, "
            "vector flat graphic design, ultra sharp, centered, no text, 8k Masterpiece"
        )
        banner_p = (
            "cinematic dark interrogation room wide angle, dramatic volumetric spotlight over metal table, "
            "faint forensic evidence board in background, moody neo-noir atmosphere, high contrast, no text, 8k"
        )
    elif any(k in sig for k in ["cosmos", "universo", "espacio", "astron", "galax", "agujero"]):
        logo_p = (
            "minimalist iconic space observatory emblem logo, glowing supermassive black hole accretion disk "
            "with event horizon geometry, deep void black background, electric cyan and gold accent, "
            "clean vector graphic, ultra sharp, centered, no text, 8k"
        )
        banner_p = (
            "breathtaking deep space nebula and supermassive black hole horizon, James Webb telescope aesthetic, "
            "cinematic widescreen cosmic scale, volumetric starlight, ultra detailed, no text, 8k"
        )
    else:
        logo_p = (
            "futuristic minimalist AI neural core emblem logo, sleek geometric cybernetic emblem glowing "
            "in neon cyan and ultraviolet on pitch black background, Apple and OpenAI aesthetic, "
            "vector graphic, ultra sharp, centered, no text, 8k"
        )
        banner_p = (
            "futuristic high-tech AI command center abstract widescreen background, glowing optical fiber "
            "neural network nodes in deep dark space, cinematic depth of field, cyan and violet accents, no text, 8k"
        )
    return logo_p, banner_p


async def generate_visual_asset(prompt: str, output_png: str, w: int, h: int) -> str:
    """Genera imágenes en alta definición usando el modelo Flux y las normaliza a PNG."""
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    raw_tmp = f"{output_png}_raw.jpg"
    seed = random.randint(1000, 999999)
    safe_prompt = urllib.parse.quote(prompt)
    url = (
        f"https://image.pollinations.ai/prompt/{safe_prompt}"
        f"?width={w}&height={h}&seed={seed}&model=flux&enhance=true&nologo=true"
    )

    async def _fetch():
        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
            res = await client.get(url)
            res.raise_for_status()
            if len(res.content) < 5000:
                raise RuntimeError("Imagen devuelta incompleta o vacía.")
            with open(raw_tmp, "wb") as f:
                f.write(res.content)

            # Convertir a PNG real y nítido (con filtro de nitidez unsharp) para Telegram y YouTube API
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_tmp,
                "-vf", f"scale={w}:{h}:flags=lanczos,unsharp=5:5:0.8:5:5:0.0",
                output_png
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return output_png

    try:
        return await retry_async(_fetch, retries=3, backoff=2.0)
    except Exception as e:
        logger.error(f"Fallo al generar asset {output_png}: {e}")
        return ""


async def send_telegram_photo(settings: Settings, photo_path: str, caption: str) -> None:
    """Envía el archivo de imagen directamente al chat de Telegram."""
    if not settings.bot_token or not settings.chat_id or not os.path.exists(photo_path):
        return
    url = f"https://api.telegram.org/bot{settings.bot_token}/sendPhoto"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(photo_path, "rb") as f:
                files = {"photo": (os.path.basename(photo_path), f, "image/png")}
                data = {
                    "chat_id": str(settings.chat_id),
                    "caption": caption[:1020],
                    "parse_mode": "HTML"
                }
                await client.post(url, data=data, files=files)
    except Exception as e:
        logger.warning(f"No se pudo enviar foto a Telegram: {e}")


async def apply_branding_and_bio_to_youtube(settings: Settings, logo_png_path: str, about_text: str, keywords_list: list[str]) -> str:
    """Reconstruye el token OAuth sin forzar scopes conflictivos y actualiza Watermark + Biografía."""
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
        except Exception as e:
            logger.warning(f"Error preparando token OAuth: {e}")

    if not os.path.exists(token_path):
        all_tokens = sorted(glob.glob("tokens/*.json"))
        if all_tokens:
            token_path = all_tokens[0]
        else:
            return "⚠️ Sin archivo token OAuth disponible."

    def _execute_updates():
        # IMPORTANTE: NO pasar lista de scopes aquí para evitar invalid_scope durante el refresh
        creds = Credentials.from_authorized_user_file(token_path)
        if not creds.valid or not creds.token:
            creds.refresh(Request())

        youtube = build("youtube", "v3", credentials=creds)
        results = []

        # Obtener el ID real del canal autenticado
        real_channel_id = settings.channel_id
        ch_item = None
        try:
            ch_res = youtube.channels().list(mine=True, part="snippet,brandingSettings").execute()
            if ch_res.get("items"):
                ch_item = ch_res["items"][0]
                real_channel_id = ch_item["id"]
        except Exception as e:
            logger.warning(f"Lectura inicial de canal: {e}")

        # 1. Subir Marca de Agua en formato PNG (requerido por YouTube)
        if logo_png_path and os.path.exists(logo_png_path):
            try:
                media = MediaFileUpload(logo_png_path, chunksize=-1, resumable=False, mimetype="image/png")
                youtube.watermarks().set(
                    channelId=real_channel_id,
                    body={
                        "timing": {
                            "type": "offsetFromStart",
                            "offsetMs": 1000,
                            "durationMs": 15000
                        },
                        "position": {
                            "type": "corner",
                            "cornerPosition": "bottomRight"
                        }
                    },
                    media_body=media
                ).execute()
                results.append("✅ Marca de Agua (Watermark PNG) instalada en los videos del canal.")
            except Exception as e:
                err_str = str(e)
                logger.error(f"Detalle Watermark API: {err_str}")
                if "insufficientPermissions" in err_str or "403" in err_str:
                    results.append("⚠️ Watermark: Tu token OAuth fue creado solo con permiso de subida. Toca '🔓 Conectar YouTube (OAuth)' en el menú para darle permiso de edición de canal.")
                else:
                    results.append(f"⚠️ Watermark no aplicado ({err_str[:90]}).")

        # 2. Actualizar Biografía SEO (Acerca de) y Keywords del Canal
        if about_text and ch_item:
            try:
                branding = ch_item.get("brandingSettings", {})
                if "channel" not in branding:
                    branding["channel"] = {}
                branding["channel"]["description"] = about_text[:980]
                if keywords_list:
                    kw_str = " ".join([f'"{k.strip()}"' for k in keywords_list[:12] if k.strip()])
                    branding["channel"]["keywords"] = kw_str[:480]

                youtube.channels().update(
                    part="brandingSettings",
                    body={
                        "id": real_channel_id,
                        "brandingSettings": branding
                    }
                ).execute()
                results.append("✅ Pestaña 'Acerca de' (Bio SEO) y Palabras Clave del canal actualizadas automáticamente.")
            except Exception as e:
                err_str = str(e)
                logger.error(f"Detalle Channels Update API: {err_str}")
                if "insufficientPermissions" in err_str or "403" in err_str:
                    results.append("⚠️ Bio SEO: Reconecta el canal desde '🔓 Conectar YouTube (OAuth)' para habilitar escritura en el perfil.")
                else:
                    results.append(f"⚠️ Bio SEO lista en texto ({err_str[:80]}).")

        return "\n".join(results) if results else "ℹ️ Activos generados."

    try:
        return await asyncio.to_thread(_execute_updates)
    except Exception as e:
        logger.error(f"Fallo OAuth en YouTube Branding: {e}")
        return f"⚠️ Token requiere re-vinculación en '🔓 Conectar YouTube (OAuth)' ({str(e)[:80]})."


async def build_channel_branding(settings: Settings) -> None:
    logger.info(f"=== INICIANDO BRANDING ENGINE PRO: {settings.channel_name} ({settings.raw_topic}) ===")

    await send_telegram(
        settings,
        f"🎨 <b>Diseñando Identidad Omnicanal HD para {escape_html(settings.channel_name)}...</b>\n"
        f"🎯 <b>Nicho:</b> <i>{escape_html(settings.raw_topic)}</i>\n\n"
        f"⚡ <i>Generando estrategia SEO, paleta HEX y renderizando Logo (1024x1024) + Banner (1920x1080) con motor Flux...</i>"
    )

    default_logo_p, default_banner_p = build_niche_visual_presets(settings.channel_name, settings.raw_topic)

    prompt_json = (
        f"Eres el Director Creativo y Especialista SEO de YouTube, TikTok, Instagram y X para el canal '{settings.channel_name}'.\n"
        f"Nicho específico del canal: '{settings.raw_topic}'.\n\n"
        "Devuelve ÚNICAMENTE un objeto JSON válido (sin bloques markdown, sin texto adicional) con esta estructura exacta:\n"
        "{\n"
        '  "paleta": ["#HEX1 (Uso)", "#HEX2 (Uso)", "#HEX3 (Uso)"],\n'
        '  "youtube_about": "Descripción SEO magnética para la pestaña Acerca de en español (300 a 500 caracteres, con palabras clave y llamado a suscribirse)",\n'
        '  "tiktok_bio": "Bio impactante para TikTok máx 80 caracteres con emojis",\n'
        '  "ig_bio": "Bio para Instagram Reels en 3 líneas con emojis y CTA",\n'
        '  "x_bio": "Bio de autoridad para X (Twitter) máx 150 caracteres",\n'
        '  "ganchos": ["Gancho 1", "Gancho 2", "Gancho 3"],\n'
        '  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5", "keyword6"],\n'
        '  "logo_prompt": "Detailed prompt in English for a high-end minimalist vector emblem icon representing specifically the channel niche, dark background, bold contrast, centered, NO TEXT, 8k",\n'
        '  "banner_prompt": "Detailed prompt in English for a cinematic widescreen 16:9 YouTube banner atmosphere representing specifically the channel niche, moody lighting, NO TEXT, 8k"\n'
        "}"
    )

    raw_llm = await ask_llm(settings, prompt_json, max_tokens=1100)

    data = {}
    try:
        clean_json = re.sub(r"```json|```", "", raw_llm).strip()
        m = re.search(r"\{[\s\S]*\}", clean_json)
        if m:
            data = json.loads(m.group(0))
    except Exception as e:
        logger.warning(f"Fallback JSON parser en Branding: {e}")

    paleta = data.get("paleta") or ["#0F172A (Fondo Abisal)", "#38BDF8 (Acento Eléctrico)", "#F43F5E (Alerta/CTR)"]
    youtube_about = str(data.get("youtube_about") or (
        f"Bienvenido a {settings.channel_name}. Exploramos a fondo {settings.raw_topic} con análisis documentales, "
        f"casos reales y revelaciones que nadie más te muestra. Suscríbete para no perderte ningún episodio."
    )).strip()
    tiktok_bio = str(data.get("tiktok_bio") or f"🔥 Lo oculto de {settings.raw_topic[:35]} 👇").strip()
    ig_bio = str(data.get("ig_bio") or f"🎬 Documentales: {settings.raw_topic[:45]}\n⚡ Nuevos episodios cada día\n👇 Guía oficial").strip()
    x_bio = str(data.get("x_bio") or f"Investigación y análisis profundo sobre {settings.raw_topic[:65]}.").strip()
    ganchos = data.get("ganchos") or [
        "Lo que estás por ver fue ocultado durante años...",
        "El 99% de las personas ignora este detalle crítico...",
        "Nadie pudo explicar lo que ocurrió en el segundo exacto..."
    ]
    keywords = data.get("keywords") or [settings.channel_name, "documental", "misterio", "casos reales", "viral"]

    raw_logo_p = str(data.get("logo_prompt") or "").strip()
    raw_banner_p = str(data.get("banner_prompt") or "").strip()

    # Combinar la directriz del LLM con los presets de calidad Flux para garantizar relación total con el canal
    logo_prompt = f"{raw_logo_p}, {default_logo_p}" if len(raw_logo_p) > 25 else default_logo_p
    banner_prompt = f"{raw_banner_p}, {default_banner_p}" if len(raw_banner_p) > 25 else default_banner_p

    os.makedirs("output/branding", exist_ok=True)
    logo_path = f"output/branding/{settings.channel_id}_logo.png"
    banner_path = f"output/branding/{settings.channel_id}_banner.png"

    logger.info("Renderizando Logo HD (1024x1024) y Banner HD (1920x1080) con modelo Flux...")
    await asyncio.gather(
        generate_visual_asset(logo_prompt, logo_path, 1024, 1024),
        generate_visual_asset(banner_prompt, banner_path, 1920, 1080)
    )

    # Enviar las imágenes directamente a Telegram para previsualización y descarga inmediata
    await send_telegram_photo(
        settings,
        logo_path,
        f"🛡️ <b>Logotipo Oficial HD (1024x1024) — {escape_html(settings.channel_name)}</b>\n"
        f"<i>Optimizado para Avatar y Marca de Agua</i>"
    )
    await send_telegram_photo(
        settings,
        banner_path,
        f"🖼️ <b>Banner Panorámico HD (1920x1080) — {escape_html(settings.channel_name)}</b>\n"
        f"<i>Optimizado para Cabecera de YouTube y X</i>"
    )

    # Aplicar en YouTube API (Watermark PNG + Bio + Keywords)
    api_status = await apply_branding_and_bio_to_youtube(settings, logo_path, youtube_about, keywords)

    paleta_str = "\n".join([f"  • <code>{escape_html(str(c))}</code>" for c in paleta])
    ganchos_str = "\n".join([f"  • <i>\"{escape_html(str(g))}\"</i>" for g in ganchos])
    kw_str = ", ".join([escape_html(str(k)) for k in keywords])

    report = (
        f"🚀 <b>BRAND KIT OMNICANAL COMPLETADO — {escape_html(settings.channel_name)}</b>\n\n"
        f"<b>Estado de Sincronización con YouTube:</b>\n{escape_html(api_status)}\n\n"
        f"🎨 <b>Paleta de Colores de Marca:</b>\n{paleta_str}\n\n"
        f"📺 <b>YouTube (Pestaña 'Acerca de'):</b>\n<pre>{escape_html(youtube_about)}</pre>\n\n"
        f"🎵 <b>TikTok Bio:</b>\n<code>{escape_html(tiktok_bio)}</code>\n\n"
        f"📸 <b>Instagram Reels Bio:</b>\n<pre>{escape_html(ig_bio)}</pre>\n\n"
        f"🐦 <b>X (Twitter) Bio:</b>\n<code>{escape_html(x_bio)}</code>\n\n"
        f"🪝 <b>Fórmulas de Gancho (Primeros 3s):</b>\n{ganchos_str}\n\n"
        f"🔑 <b>Keywords SEO del Canal:</b>\n<code>{kw_str}</code>"
    )

    await send_telegram(settings, report)
    logger.info("=== BRANDING ENGINE FINALIZADO CON ÉXITO ===")
