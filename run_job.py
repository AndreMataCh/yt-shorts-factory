import os
import re
import glob
import json
import asyncio
import subprocess
import urllib.parse
import httpx
import numpy as np
import soundfile as sf
from kokoro import KPipeline
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

def clean_secret(val: str) -> str:
    if not val:
        return ""
    # Elimina comillas, saltos de línea y espacios accidentales al copiar/pegar en GitHub Secrets
    return re.sub(r'[\r\n\t"\'\s]+', '', val)

# === VARIABLES DE ENTORNO Y PAYLOAD DE PAULA / WORKER ===
BOT_TOKEN = clean_secret(os.getenv("TELEGRAM_BOT_TOKEN") or "")
GROQ_API_KEY = clean_secret(os.getenv("GROQ_API_KEY") or "")
GEMINI_API_KEY = clean_secret(os.getenv("GEMINI_API_KEY") or "")
PEXELS_API_KEY = clean_secret(os.getenv("PEXELS_API_KEY") or "")

CHANNEL_ID = (os.getenv("CHANNEL_ID") or "ch1").strip().lower()
CHANNEL_NAME = (os.getenv("CHANNEL_NAME") or CHANNEL_ID.upper()).strip()
CHANNEL_TOKEN_JSON = (os.getenv("CHANNEL_TOKEN_JSON") or "").strip()

RAW_TOPIC = (os.getenv("TOPIC") or "Los secretos mejor guardados de la tecnología").strip()
VISUAL_STYLE = (os.getenv("VISUAL_STYLE") or "1").strip()
VOICE_STYLE = (os.getenv("VOICE_STYLE") or "1").strip()
RAW_COUNT = (os.getenv("VIDEO_COUNT") or "1").strip()
MODE_CHOICE = (os.getenv("MODE") or "3").strip()

MONETIZATION_URL = (os.getenv("MONETIZATION_URL") or "").strip()
CUSTOM_TITLE = (os.getenv("CUSTOM_TITLE") or "").strip()
CUSTOM_DESCRIPTION = (os.getenv("CUSTOM_DESCRIPTION") or "").strip()
SOURCE_VIDEO_URL = (os.getenv("SOURCE_VIDEO_URL") or "").strip()
CHAT_ID = clean_secret(os.getenv("CHAT_ID") or "")

try:
    VIDEO_COUNT = max(1, min(int(re.sub(r"[^0-9]", "", RAW_COUNT) or "1"), 10))
except ValueError:
    VIDEO_COUNT = 1

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}

# Materializar el token enviado desde Cloudflare KV (Arquitectura PAULA)
os.makedirs("tokens", exist_ok=True)
if CHANNEL_TOKEN_JSON:
    try:
        parsed_tok = json.loads(CHANNEL_TOKEN_JSON)
        with open(f"tokens/{CHANNEL_ID}_token_kv.json", "w", encoding="utf-8") as f:
            json.dump(parsed_tok, f)
        print(f"Token OAuth cargado desde Cloudflare KV para el canal: {CHANNEL_NAME} ({CHANNEL_ID})")
    except Exception as e:
        print(f"Aviso al parsear CHANNEL_TOKEN_JSON: {e}")

print("Inicializando motor neuronal Kokoro-82M...")
kokoro_pipeline = KPipeline(lang_code='e')

async def send_telegram(text: str):
    print(text)
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, json={"chat_id": int(CHAT_ID), "text": text, "parse_mode": "HTML"})
        except Exception as e:
            print(f"Aviso Telegram: {e}")

def clean_llm_output(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()

async def ask_llm(prompt: str, max_tokens: int = 1200) -> str:
    errors = []

    # 1. GOOGLE GEMINI CON AUTO-DESCUBRIMIENTO DE MODELOS ACTIVOS
    if GEMINI_API_KEY and GEMINI_API_KEY.startswith("AIza"):
        gemini_headers = {
            **BROWSER_HEADERS,
            "x-goog-api-key": GEMINI_API_KEY,
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=60.0, headers=gemini_headers) as client:
            models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest", "gemini-pro-latest"]
            try:
                list_res = await client.get("https://generativelanguage.googleapis.com/v1beta/models")
                if list_res.status_code == 200:
                    avail = [
                        m["name"].replace("models/", "")
                        for m in list_res.json().get("models", [])
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                        and "gemini" in m.get("name", "")
                        and "vision" not in m.get("name", "")
                        and "embedding" not in m.get("name", "")
                    ]
                    # Priorizar modelos flash disponibles en la cuenta
                    flash_First = sorted(avail, key=lambda x: (0 if "flash" in x else 1))
                    if flash_First:
                        models_to_try = flash_First[:4]
            except Exception:
                pass

            for model in models_to_try:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                    res = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
                    if res.status_code == 200:
                        candidates = res.json().get("candidates", [])
                        if candidates:
                            content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                            cleaned = clean_llm_output(content)
                            if cleaned:
                                return cleaned
                    else:
                        errors.append(f"Gemini ({model}): HTTP {res.status_code}")
                except Exception as e:
                    errors.append(f"Gemini ({model}): {str(e)[:60]}")
    elif GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY inválida (no inicia con AIza)")

    # 2. GROQ CON AUTO-DESCUBRIMIENTO DE MODELOS ACTIVOS
    if GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_"):
        groq_headers = {
            **BROWSER_HEADERS,
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=60.0, headers=groq_headers) as client:
            groq_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"]
            try:
                m_res = await client.get("https://api.groq.com/openai/v1/models")
                if m_res.status_code == 200:
                    avail_g = [
                        m["id"] for m in m_res.json().get("data", [])
                        if "whisper" not in m["id"].lower() and "guard" not in m["id"].lower() and "tts" not in m["id"].lower()
                    ]
                    if avail_g:
                        # Priorizar llama-3.3 o llama-3.1 si existen
                        avail_g.sort(key=lambda x: (0 if "llama-3.3" in x else (1 if "llama" in x else 2)))
                        groq_models = avail_g[:4]
            except Exception:
                pass

            for model in groq_models:
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
                            return cleaned
                    else:
                        errors.append(f"Groq ({model}): HTTP {res.status_code}")
                except Exception as e:
                    errors.append(f"Groq ({model}): {str(e)[:60]}")
    elif GROQ_API_KEY:
        errors.append("GROQ_API_KEY inválida (no inicia con gsk_)")

    # 3. RESPALDO GRATUITO OFICIAL 1: POLLINATIONS OPENAI ENDPOINT (/openai)
    async with httpx.AsyncClient(timeout=60.0, headers={**BROWSER_HEADERS, "Content-Type": "application/json"}, follow_redirects=True) as client:
        for poly_model in ["openai", "mistral", "llama"]:
            try:
                res = await client.post(
                    "https://text.pollinations.ai/openai",
                    json={
                        "model": poly_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7
                    }
                )
                if res.status_code == 200:
                    data = res.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0].get("message", {}).get("content", "")
                        cleaned = clean_llm_output(content)
                        if cleaned:
                            return cleaned
                else:
                    errors.append(f"Pollinations OpenAI ({poly_model}): HTTP {res.status_code}")
            except Exception as e:
                errors.append(f"Pollinations OpenAI ({poly_model}): {str(e)[:60]}")

    # 4. RESPALDO GRATUITO 2: POLLINATIONS ROOT POST Y GET
    async with httpx.AsyncClient(timeout=60.0, headers={**BROWSER_HEADERS, "Content-Type": "application/json"}, follow_redirects=True) as client:
        try:
            res = await client.post(
                "https://text.pollinations.ai/",
                json={"messages": [{"role": "user", "content": prompt}], "model": "openai"}
            )
            if res.status_code == 200 and len(res.text.strip()) > 10 and "<!DOCTYPE" not in res.text:
                cleaned = clean_llm_output(res.text)
                if cleaned:
                    return cleaned
        except Exception as e:
            errors.append(f"Pollinations Root: {str(e)[:60]}")

    async with httpx.AsyncClient(timeout=45.0, headers=BROWSER_HEADERS, follow_redirects=True) as client:
        try:
            encoded_p = urllib.parse.quote(prompt[:600])
            res = await client.get(f"https://text.pollinations.ai/{encoded_p}")
            if res.status_code == 200 and len(res.text.strip()) > 5 and "<!DOCTYPE" not in res.text:
                cleaned = clean_llm_output(res.text)
                if cleaned:
                    return cleaned
        except Exception as e:
            errors.append(f"Pollinations GET: {str(e)[:60]}")

    raise RuntimeError(f"Falla de LLMs: {' | '.join(errors)[:350]}")

async def generate_long_plan(topic: str) -> dict:
    script_prompt = (
        f"Escribe un guion documental fascinante y dinámico en español neutro de unas 350 a 450 palabras sobre: '{topic}'. "
        "Estructura: Gancho impactante al inicio, desarrollo con alto ritmo narrativo y conclusión memorable. "
        "REGLAS: Devuelve ÚNICAMENTE el texto hablado. Sin asteriscos, sin títulos, sin emojis."
    )
    script = re.sub(r"[\*\#\[\]\(\)]", "", await ask_llm(script_prompt, max_tokens=1400)).strip()
    title = CUSTOM_TITLE or (await ask_llm(f"Crea un título magnético de YouTube (máx 65 caracteres) sobre '{topic}'. SOLO el título sin comillas.", max_tokens=150)).replace('"', '').splitlines()[0].strip()[:75]

    terms_raw = await ask_llm(
        f"For the topic '{topic}', provide exactly 10 distinct, dynamic cinematic video search keywords in English. Return ONLY the 10 comma-separated terms.",
        max_tokens=300
    )
    terms = [t.strip() for t in terms_raw.split(",") if t.strip()][:10]
    while len(terms) < 10:
        terms.append("cinematic abstract motion background")

    monetization_block = f"\n\n🔥 Recursos y enlaces exclusivos: {MONETIZATION_URL}" if MONETIZATION_URL else ""
    base_desc = CUSTOM_DESCRIPTION or f"{title}\n\nExploramos a fondo: {topic}.\n\n{script[:250]}..."

    return {
        "title": title,
        "description": f"{base_desc}{monetization_block}\n\n#Documental #Tecnologia #Viral",
        "tags": ["documental", "tecnologia", "ciencia", "viral"],
        "script": script,
        "search_terms": terms
    }

async def generate_teaser_short_plan(topic: str, long_title: str = "", long_url: str = "") -> dict:
    cta_line = "Descubre el video completo en el enlace de la descripción." if long_url else "Suscríbete para descubrir más secretos como este."
    script_prompt = (
        f"Escribe un guion de YouTube Short ultradinámico de 30 segundos sobre '{topic}'. "
        f"Empieza con un gancho agresivo y termina diciendo: '{cta_line}' "
        "REGLAS: Devuelve ÚNICAMENTE el texto hablado en español neutro, sin asteriscos ni emojis."
    )
    script = re.sub(r"[\*\#\[\]\(\)]", "", await ask_llm(script_prompt, max_tokens=600)).strip()
    title = (await ask_llm(f"Crea un título corto y viral (máx 55 caracteres) para un Short sobre '{topic}'. SOLO el título sin comillas.", max_tokens=150)).replace('"', '').splitlines()[0].strip()[:65]

    terms_raw = await ask_llm(
        f"Provide 6 high-speed dynamic video search keywords in English about '{topic}'. Return ONLY the 6 comma-separated terms.",
        max_tokens=200
    )
    terms = [t.strip() for t in terms_raw.split(",") if t.strip()][:6]
    while len(terms) < 6:
        terms.append("fast paced abstract motion")

    desc_link = f"\n\n🔗 Video completo: {long_url}" if long_url else ""
    monetization_block = f"\n🔥 Enlace destacado: {MONETIZATION_URL}" if MONETIZATION_URL else ""
    return {
        "title": title,
        "description": f"{title}{desc_link}{monetization_block}\n\n#Shorts #Tech",
        "tags": ["shorts", "tecnologia"],
        "script": script,
        "search_terms": terms
    }

def format_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def get_audio_duration(audio_path: str) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ]).decode().strip()
    return float(out)

async def generate_voice_kokoro(script_text: str, audio_path: str, is_short: bool = True):
    voice_map = {"1": "em_alex", "2": "ef_dora", "3": "em_santa"}
    voice_name = voice_map.get(VOICE_STYLE, "em_alex")
    speed_val = 1.10 if is_short else 1.03

    generator = kokoro_pipeline(script_text, voice=voice_name, speed=speed_val)
    audio_chunks = [chunk for _, _, chunk in generator]

    if audio_chunks:
        complete_audio = np.concatenate(audio_chunks)
        sf.write(audio_path, complete_audio, 24000)
    else:
        raise RuntimeError("Kokoro no generó audio.")

async def prepare_background_music(duration: float, bgm_path: str):
    bgm_urls = [
        "https://cdn.pixabay.com/download/audio/2022/05/27/audio_1808fbf07a.mp3",
        "https://cdn.pixabay.com/download/audio/2022/01/18/audio_d0a13f69d2.mp3"
    ]
    raw_bgm = f"{bgm_path}_raw.mp3"
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
        for url in bgm_urls:
            try:
                res = await client.get(url)
                if res.status_code == 200 and len(res.content) > 20000:
                    with open(raw_bgm, "wb") as f:
                        f.write(res.content)
                    subprocess.run([
                        "ffmpeg", "-y", "-stream_loop", "-1", "-t", str(duration),
                        "-i", raw_bgm, "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", bgm_path
                    ], check=True)
                    return
            except Exception:
                pass

    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"aevalsrc='0.12*sin(2*PI*130.81*t)+0.08*sin(2*PI*196*t)':s=24000:d={duration}",
        "-af", "lowpass=f=600", "-c:a", "pcm_s16le", bgm_path
    ], check=True)

def postproducir_video_maestro(input_video: str, output_video: str, is_short: bool = True):
    w, h = (1080, 1920) if is_short else (1920, 1080)
    filtro_audio = "silenceremove=stop_periods=-1:stop_duration=0.4:stop_threshold=-50dB,loudnorm=I=-16:TP=-1.5:LRA=11,acompressor"
    filtro_video = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps=30"

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-af", filtro_audio,
        "-vf", filtro_video,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        output_video
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_video

async def edit_uploaded_raw_video(video_url: str, prefix: str, is_short: bool = True) -> str:
    os.makedirs("output", exist_ok=True)
    raw_input = f"output/{prefix}_uploaded_raw.mp4"
    final_video = f"output/{prefix}_edited_pro.mp4"
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
        res = await client.get(video_url)
        res.raise_for_status()
        with open(raw_input, "wb") as f:
            f.write(res.content)
    return postproducir_video_maestro(raw_input, final_video, is_short=is_short)

async def render_video_with_real_clips(prefix: str, script_text: str, search_terms: list[str], is_short: bool = True) -> str:
    os.makedirs("output", exist_ok=True)
    w, h = (1080, 1920) if is_short else (1920, 1080)
    orientation = "portrait" if is_short else "landscape"

    voice_path = f"output/{prefix}_voice.wav"
    bgm_path = f"output/{prefix}_bgm.wav"
    mixed_audio = f"output/{prefix}_master.wav"
    ass_path = f"output/{prefix}_subs.ass"
    raw_output = f"output/{prefix}_raw.mp4"
    final_video = f"output/{prefix}_final.mp4"

    await generate_voice_kokoro(script_text, voice_path, is_short=is_short)
    total_duration = get_audio_duration(voice_path)

    await prepare_background_music(total_duration, bgm_path)
    fade_start = max(0.0, total_duration - 2.0)

    subprocess.run([
        "ffmpeg", "-y", "-i", voice_path, "-i", bgm_path,
        "-filter_complex",
        f"[0:a]equalizer=f=1000:t=q:w=1:g=3[v];[1:a]volume=0.10,afade=t=out:st={fade_start}:d=2[b];[v][b]amix=inputs=2:duration=first[aout]",
        "-map", "[aout]", "-c:a", "aac", "-b:a", "192k", mixed_audio
    ], check=True)

    words = script_text.split()
    time_per_word = total_duration / max(len(words), 1)
    font_size = 92 if is_short else 65
    margin_v = 400 if is_short else 80

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ProSub,Arial Black,{font_size},&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,3,2,30,30,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header)
        chunk_size = 2 if is_short else 4
        for idx in range(0, len(words), chunk_size):
            phrase = " ".join(words[idx:idx+chunk_size]).upper()
            start = format_ass_time(idx * time_per_word)
            end = format_ass_time(min((idx + chunk_size) * time_per_word, total_duration))
            f.write(f"Dialogue: 0,{start},{end},ProSub,,0,0,0,,{phrase}\n")

    clip_paths = []
    clip_duration = max(total_duration / len(search_terms), 3.0)

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
        for i, term in enumerate(search_terms):
            clip_file = f"output/{prefix}_clip_{i}.mp4"
            got_clip = False
            if PEXELS_API_KEY:
                try:
                    pexels_url = f"https://api.pexels.com/videos/search?query={urllib.parse.quote(term)}&orientation={orientation}&per_page=5"
                    r = await client.get(pexels_url, headers={"Authorization": PEXELS_API_KEY})
                    videos = r.json().get("videos", [])
                    if videos:
                        chosen = np.random.choice(videos)
                        vfiles = chosen.get("video_files", [])
                        if vfiles:
                            best = max(vfiles, key=lambda x: x.get("width", 0))
                            vid_data = await client.get(best["link"])
                            raw_vid = f"output/{prefix}_source_{i}.mp4"
                            with open(raw_vid, "wb") as f:
                                f.write(vid_data.content)
                            subprocess.run([
                                "ffmpeg", "-y", "-stream_loop", "-1", "-t", str(clip_duration), "-i", raw_vid,
                                "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps=30",
                                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", clip_file
                            ], check=True)
                            got_clip = True
                except Exception:
                    pass

            if not got_clip:
                subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=0x0f172a:s={w}x{h}:d={clip_duration}:r=30",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", clip_file
                ], check=True)

            clip_paths.append(str(os.path.abspath(clip_file)))

    concat_file = f"output/{prefix}_concat.txt"
    with open(concat_file, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{cp}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
        "-i", mixed_audio, "-vf", f"ass={ass_path}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-shortest", raw_output
    ], check=True)

    postproducir_video_maestro(raw_output, final_video, is_short=is_short)
    return final_video

def upload_youtube_rotated(video_path: str, title: str, description: str, tags: list[str], is_short: bool = True, publish_at: str = None) -> str:
    channel_tokens = sorted(glob.glob(f"tokens/{CHANNEL_ID}_token_*.json")) or sorted(glob.glob("tokens/*.json"))
    if not channel_tokens:
        raise RuntimeError(f"No se encontraron tokens para el canal '{CHANNEL_NAME}' ({CHANNEL_ID}). Conéctalo desde el botón OAuth de Telegram.")

    status_body = {"selfDeclaredMadeForKids": False}
    if publish_at:
        status_body["privacyStatus"] = "private"
        status_body["publishAt"] = publish_at
    else:
        status_body["privacyStatus"] = "public"

    for token_path in channel_tokens:
        try:
            creds = Credentials.from_authorized_user_file(token_path, ["https://www.googleapis.com/auth/youtube.upload"])
            youtube = build("youtube", "v3", credentials=creds)
            body = {
                "snippet": {"title": title[:100], "description": description, "tags": tags, "categoryId": "27"},
                "status": status_body
            }
            media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
            resp = youtube.videos().insert(part="snippet,status", body=body, media_body=media).execute()
            vid_id = resp["id"]
            return f"https://youtube.com/shorts/{vid_id}" if is_short else f"https://youtu.be/{vid_id}"
        except HttpError as e:
            if e.resp.status in [403, 429]:
                continue
            raise e
    raise RuntimeError("Cuota agotada en los tokens de YouTube disponibles.")

async def main():
    try:
        await send_telegram(f"🚀 <b>Iniciando Fábrica Pro v4.0</b>\n📺 <b>Canal:</b> {CHANNEL_NAME}\n📌 <b>Tarea:</b> {RAW_TOPIC}")

        if SOURCE_VIDEO_URL:
            is_short_mode = (MODE_CHOICE == "1")
            edited_vid = await edit_uploaded_raw_video(SOURCE_VIDEO_URL, "raw_edit", is_short=is_short_mode)
            title = CUSTOM_TITLE or RAW_TOPIC[:80]
            desc = CUSTOM_DESCRIPTION or f"{title}\n\n{MONETIZATION_URL}"
            vid_url = upload_youtube_rotated(edited_vid, title, desc, ["viral", "pro"], is_short=is_short_mode)
            await send_telegram(f"✅ <b>Video Pregrabado Masterizado y Publicado:</b>\n🔗 {vid_url}")
            return

        subtopics = [RAW_TOPIC]
        for idx, topic in enumerate(subtopics, start=1):
            long_url = ""
            long_title = topic

            if MODE_CHOICE in ["2", "3"]:
                long_plan = await generate_long_plan(topic)
                long_title = long_plan["title"]
                long_vid = await render_video_with_real_clips(f"long_{idx}", long_plan["script"], long_plan["search_terms"], is_short=False)
                long_url = upload_youtube_rotated(long_vid, long_plan["title"], long_plan["description"], long_plan["tags"], is_short=False)
                await send_telegram(f"✅ <b>Video Largo Publicado en {CHANNEL_NAME}:</b>\n📺 {long_plan['title']}\n🔗 {long_url}")

            if MODE_CHOICE in ["1", "3"]:
                short_plan = await generate_teaser_short_plan(topic, long_title, long_url)
                short_vid = await render_video_with_real_clips(f"short_{idx}", short_plan["script"], short_plan["search_terms"], is_short=True)
                short_url = upload_youtube_rotated(short_vid, short_plan["title"], short_plan["description"], short_plan["tags"], is_short=True)
                await send_telegram(f"🔥 <b>Short Dinámico Publicado en {CHANNEL_NAME}:</b>\n🔗 {short_url}")

        await send_telegram(f"🏁 <b>¡Tanda finalizada con éxito en {CHANNEL_NAME}!</b>")
    except Exception as e:
        await send_telegram(f"❌ <b>Error crítico en fábrica ({CHANNEL_NAME}):</b>\n<code>{str(e)}</code>")
        raise e

if __name__ == "__main__":
    asyncio.run(main())
