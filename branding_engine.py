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
from googleapiclient.errors import HttpError

from config import Settings
from llm import ask_llm
from utils import send_telegram, retry_async, BROWSER_HEADERS, escape_html

logger = logging.getLogger("PAULA.Branding")


async def download_custom_logo(logo_url: str, output_path: str) -> str:
    """Descarga el logo enviado por el usuario desde Telegram o la Web."""
    if not logo_url:
        return ""
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
            res = await client.get(logo_url)
            res.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(res.content)
            logger.info(f"Logo personalizado descargado exitosamente en {output_path}")
            return output_path
    except Exception as e:
        logger.error(f"Fallo al descargar logo personalizado desde {logo_url}: {e}")
        return ""


async def generate_visual_asset(prompt: str, output_png: str, w: int, h: int, reference_img: str = "") -> str:
    """Genera activos visuales (o adapta el Banner) en alta definición usando modelo Flux."""
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


def prepare_youtube_compliant_assets(logo_png: str, banner_png: str) -> tuple[str, str]:
    """Prepara la marca de agua PNG RGBA 400x400 y el Banner JPG 2048x1152 para la API de YouTube."""
    watermark_rgba_png = logo_png.replace(".png", "_yt_watermark.png")
    banner_2048_jpg = banner_png.replace(".png", "_yt_banner_2048.jpg")

    if os.path.exists(logo_png):
        try:
            geq_alpha_expr = (
                "r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
                "a='if(gt(pow(max(0,abs(X-200)-140),2)+pow(max(0,abs(Y-200)-140),2),3600),0,235)'"
            )
            subprocess.run([
                "ffmpeg", "-y", "-i", logo_png,
                "-vf", f"scale=400:400:flags=lanczos,format=rgba,geq={geq_alpha_expr}",
                "-pix_fmt", "rgba",
                watermark_rgba_png
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except Exception:
            subprocess.run([
                "ffmpeg", "-y", "-i", logo_png,
                "-vf", "scale=400:400,format=rgba",
                "-pix_fmt", "rgba",
                watermark_rgba_png
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    if os.path.exists(banner_png):
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", banner_png,
                "-vf", "scale=2048:1152:flags=lanczos",
                "-q:v", "2",
                banner_2048_jpg
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except Exception:
            pass

    return watermark_rgba_png, banner_2048_jpg


def extract_youtube_error_reason(e: Exception) -> str:
    if isinstance(e, HttpError):
        try:
            err_data = json.loads(e.content.decode("utf-8", errors="ignore"))
            msg = err_data.get("error", {}).get("message", "")
            reasons = [err.get("reason", "") for err in err_data.get("error", {}).get("errors", [])]
            return f"HTTP {e.resp.status}: {msg} ({', '.join(reasons)})"
        except Exception:
            return f"HTTP {e.resp.status}: {str(e)[:140]}"
    return str(e)[:140]


async def send_telegram_photo(settings: Settings, photo_path: str, caption: str) -> None:
    if not settings.bot_token or not settings.chat_id or not os.path.exists(photo_path):
        return
    url = f"https://api.telegram.org/bot{settings.bot_token}/sendPhoto"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(photo_path, "rb") as f:
                files = {"photo": (os.path.basename(photo_path), f, "image/png")}
                data = {"chat_id": str(settings.chat_id), "caption": caption[:1020], "parse_mode": "HTML"}
                await client.post(url, data=data, files=files)
    except Exception as e:
        logger.warning(f"No se pudo enviar foto a Telegram: {e}")


async def apply_branding_and_bio_to_youtube(
    settings: Settings,
    logo_png_path: str,
    banner_png_path: str,
    about_text: str,
    keywords_list: list[str]
) -> str:
    """Aplica de forma 100% automatizada el Watermark, el Banner, la Biografía y las Keywords en YouTube."""
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

    watermark_rgba, banner_2048 = prepare_youtube_compliant_assets(logo_png_path, banner_png_path)

    def _execute_updates():
        creds = Credentials.from_authorized_user_file(token_path)
        if not creds.valid or not creds.token:
            creds.refresh(Request())

        youtube = build("youtube", "v3", credentials=creds)
        results = []

        real_channel_id = settings.channel_id
        ch_item = None
        try:
            ch_res = youtube.channels().list(mine=True, part="snippet,brandingSettings").execute()
            if ch_res.get("items"):
                ch_item = ch_res["items"][0]
                real_channel_id = ch_item["id"]
        except Exception as e:
            logger.warning(f"Lectura inicial de canal: {extract_youtube_error_reason(e)}")

        # 1. Aplicar Marca de Agua (Watermark PNG RGBA)
        wm_target = watermark_rgba if os.path.exists(watermark_rgba) else logo_png_path
        if wm_target and os.path.exists(wm_target):
            try:
                media = MediaFileUpload(wm_target, mimetype="image/png", resumable=True)
                youtube.watermarks().set(
                    channelId=real_channel_id,
                    body={"timing": {"type": "offsetFromStart", "offsetMs": 0}, "position": {"type": "corner", "cornerPosition": "bottomRight"}},
                    media_body=media
                ).execute()
                results.append("✅ Marca de Agua (Watermark PNG) instalada automáticamente en los videos.")
            except Exception as e:
                reason = extract_youtube_error_reason(e)
                results.append(f"⚠️ Watermark no aplicado ({reason}).")

        # 2. Subir Banner Panorámico 2048x1152
        banner_external_url = ""
        if os.path.exists(banner_2048):
            try:
                banner_media = MediaFileUpload(banner_2048, mimetype="image/jpeg", resumable=True)
                banner_res = youtube.channelBanners().insert(body={}, media_body=banner_media).execute()
                banner_external_url = banner_res.get("url", "")
            except Exception as e:
                reason = extract_youtube_error_reason(e)
                results.append(f"⚠️ Subida de Banner rechazada ({reason}).")

        # 3. Actualizar brandingSettings (Bio SEO, Keywords y Banner URL)
        if ch_item:
            try:
                branding = ch_item.get("brandingSettings", {})
                if "channel" not in branding:
                    branding["channel"] = {}
                if about_text:
                    branding["channel"]["description"] = about_text[:980]
                if keywords_list:
                    kw_clean = [re.sub(r'["\'\r\n]+', '', str(k)).strip() for k in keywords_list[:10] if str(k).strip()]
                    kw_str = " ".join([f'"{k}"' if " " in k else k for k in kw_clean])
                    branding["channel"]["keywords"] = kw_str[:480]

                if banner_external_url:
                    if "image" not in branding:
                        branding["image"] = {}
                    branding["image"]["bannerExternalUrl"] = banner_external_url

                youtube.channels().update(
                    part="brandingSettings",
                    body={"id": real_channel_id, "brandingSettings": branding}
                ).execute()

                if banner_external_url:
                    results.append("✅ Banner de cabecera, Biografía 'Acerca de' y Keywords aplicados en YouTube.")
                else:
                    results.append("✅ Pestaña 'Acerca de' (Bio SEO) y Keywords aplicados en YouTube.")
            except Exception as e:
                reason = extract_youtube_error_reason(e)
                results.append(f"⚠️ Actualización de perfil rechazada ({reason}).")

        return "\n".join(results) if results else "ℹ️ Activos aplicados."

    try:
        return await asyncio.to_thread(_execute_updates)
    except Exception as e:
        return f"⚠️ Error OAuth: {str(e)[:90]}"


async def build_channel_branding(settings: Settings) -> None:
    logger.info(f"=== INICIANDO BRANDING ENGINE PRO: {settings.channel_name} ({settings.raw_topic}) ===")

    logo_path = f"output/branding/{settings.channel_id}_logo.png"
    banner_path = f"output/branding/{settings.channel_id}_banner.png"
    os.makedirs("output/branding", exist_ok=True)

    # Verificar si el usuario envió un logo personalizado mediante BUNDLE_JSON
    custom_logo_url = getattr(settings, "custom_logo_url", "")
    if custom_logo_url:
        await send_telegram(settings, f"📥 <i>Descargando logo corporativo enviado por el operador...</i>")
        downloaded = await download_custom_logo(custom_logo_url, logo_path)
        if downloaded:
            # Normalizar logo personalizado a 1024x1024 con fondo limpio
            subprocess.run(["ffmpeg", "-y", "-i", logo_path, "-vf", "scale=1024:1024:flags=lanczos", logo_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    has_custom_logo = os.path.exists(logo_path) and os.path.getsize(logo_path) > 1000

    await send_telegram(
        settings,
        f"🎨 <b>Diseñando Brand Kit Omnicanal para {escape_html(settings.channel_name)}...</b>\n"
        f"🎯 <b>Nicho:</b> <i>{escape_html(settings.raw_topic)}</i>\n"
        f"🛡️ <b>Logotipo:</b> {'<code>[Personalizado por Usuario]</code>' if has_custom_logo else '<code>[Generado por IA Flux]</code>'}\n\n"
        f"⚡ <i>Extrayendo paleta cromática, generando biografía SEO, ganchos de retención y Banner HD (2048x1152)...</i>"
    )

    prompt_json = (
        f"Eres el Director Creativo y Especialista SEO de YouTube, TikTok, Instagram y X para el canal '{settings.channel_name}'.\n"
        f"Nicho específico: '{settings.raw_topic}'.\n"
        f"¿Tiene el canal un logo corporativo previo proporcionado?: {'Sí' if has_custom_logo else 'No (crear concepto nuevo)'}.\n\n"
        "Devuelve ÚNICAMENTE un objeto JSON válido (sin bloques markdown, sin texto adicional) con esta estructura exacta:\n"
        "{\n"
        '  "paleta": ["#HEX1 (Uso principal)", "#HEX2 (Acento)", "#HEX3 (Contraste)"],\n'
        '  "youtube_about": "Descripción SEO magnética para la pestaña Acerca de en español (300 a 500 caracteres, con palabras clave de alto tráfico)",\n'
        '  "tiktok_bio": "Bio impactante para TikTok máx 80 caracteres con emojis",\n'
        '  "ig_bio": "Bio para Instagram Reels en 3 líneas con emojis y CTA",\n'
        '  "x_bio": "Bio de autoridad para X (Twitter) máx 150 caracteres",\n'
        '  "ganchos": ["Gancho 1", "Gancho 2", "Gancho 3"],\n'
        '  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],\n'
        '  "banner_prompt": "Detailed prompt in English for a cinematic widescreen 16:9 YouTube banner atmosphere matching the channel niche and color harmony, moody lighting, NO TEXT, 8k"\n'
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

    paleta = data.get("paleta") or ["#0F172A", "#38BDF8", "#F43F5E"]
    youtube_about = str(data.get("youtube_about") or f"Bienvenido a {settings.channel_name}. Exploramos a fondo {settings.raw_topic}. Suscríbete.").strip()
    tiktok_bio = str(data.get("tiktok_bio") or f"🔥 Lo mejor de {settings.raw_topic[:30]} 👇").strip()
    ig_bio = str(data.get("ig_bio") or f"🎬 {settings.raw_topic[:40]}\n⚡ Contenido diario\n👇 Guía oficial").strip()
    x_bio = str(data.get("x_bio") or f"Análisis profundo sobre {settings.raw_topic[:60]}.").strip()
    ganchos = data.get("ganchos") or ["Lo que estás por ver fue ocultado...", "El 99% de las personas ignora esto...", "Nadie pudo explicar lo que ocurrió..."]
    keywords = data.get("keywords") or [settings.channel_name, "documental", "misterio", "casos reales"]
    banner_prompt = str(data.get("banner_prompt") or f"cinematic abstract banner representing {settings.raw_topic}, dark moody lighting, 8k").strip()

    # Si no hay logo personalizado, generar uno nuevo con IA Flux
    if not has_custom_logo:
        logo_prompt = f"minimalist vector emblem icon representing {settings.raw_topic}, dark background, bold contrast, centered, NO TEXT, 8k"
        await generate_visual_asset(logo_prompt, logo_path, 1024, 1024)

    # Renderizar Banner HD
    await generate_visual_asset(banner_prompt, banner_path, 1920, 1080)

    # Enviar previsualizaciones a Telegram
    if os.path.exists(logo_path):
        await send_telegram_photo(settings, logo_path, f"🛡️ <b>Logotipo del Canal — {escape_html(settings.channel_name)}</b>")
    if os.path.exists(banner_path):
        await send_telegram_photo(settings, banner_path, f"🖼️ <b>Banner Panorámico HD — {escape_html(settings.channel_name)}</b>")

    # Aplicar cambios en YouTube de forma automatizada
    api_status = await apply_branding_and_bio_to_youtube(settings, logo_path, banner_path, youtube_about, keywords)

    paleta_str = "\n".join([f"  • <code>{escape_html(str(c))}</code>" for c in paleta])
    ganchos_str = "\n".join([f"  • <i>\"{escape_html(str(g))}\"</i>" for g in ganchos])
    kw_str = ", ".join([escape_html(str(k)) for k in keywords])

    report = (
        f"🚀 <b>BRAND KIT OMNICANAL APLICADO — {escape_html(settings.channel_name)}</b>\n\n"
        f"<b>Sincronización Automática con YouTube:</b>\n{escape_html(api_status)}\n\n"
        f"🎨 <b>Paleta Cromática Extraída:</b>\n{paleta_str}\n\n"
        f"📺 <b>YouTube (Acerca de):</b>\n<pre>{escape_html(youtube_about)}</pre>\n\n"
        f"🎵 <b>TikTok Bio:</b> <code>{escape_html(tiktok_bio)}</code>\n"
        f"📸 <b>IG Bio:</b> <pre>{escape_html(ig_bio)}</pre>\n"
        f"🐦 <b>X Bio:</b> <code>{escape_html(x_bio)}</code>\n\n"
        f"🪝 <b>Ganchos Virales (3s):</b>\n{ganchos_str}\n\n"
        f"🔑 <b>Keywords SEO:</b> <code>{kw_str}</code>"
    )

    await send_telegram(settings, report)
    logger.info("=== BRANDING OMNICANAL APLICADO CON ÉXITO ===")
