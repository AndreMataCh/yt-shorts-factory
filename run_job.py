import os
import re
import glob
import asyncio
import subprocess
import urllib.parse
import httpx
import edge_tts
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
PEXELS_API_KEY = (os.getenv("PEXELS_API_KEY") or "").strip()
RAW_TOPIC = (os.getenv("TOPIC") or "Las 3 paradojas más extrañas del tiempo").strip()
CHAT_ID = (os.getenv("CHAT_ID") or "1008445368").strip()

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}

ACTIVE_GROQ_MODEL = None

async def send_telegram(text: str):
    print(text)
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, json={"chat_id": int(CHAT_ID), "text": text})
        except Exception as e:
            print(f"Aviso Telegram: {e}")

# === MOTOR LLM CON AUTO-DETECCIÓN DE MODELOS ===
async def get_working_groq_model(client: httpx.AsyncClient, headers: dict) -> str:
    global ACTIVE_GROQ_MODEL
    if ACTIVE_GROQ_MODEL:
        return ACTIVE_GROQ_MODEL
    res = await client.get("https://api.groq.com/openai/v1/models", headers=headers)
    if res.status_code != 200:
        raise RuntimeError(f"Error Groq Models ({res.status_code}): {res.text}")
    all_models = [m["id"] for m in res.json().get("data", [])]
    chat_models = [
        m for m in all_models
        if not any(x in m.lower() for x in ["whisper", "guard", "tts", "playai", "vision", "embed", "orpheus"])
    ]
    for keyword in ["70b", "llama-4", "qwen", "deepseek", "llama-3", "llama3", "gemma", "8b"]:
        for m in chat_models:
            if keyword in m.lower():
                ACTIVE_GROQ_MODEL = m
                return ACTIVE_GROQ_MODEL
    ACTIVE_GROQ_MODEL = chat_models[0]
    return ACTIVE_GROQ_MODEL

async def ask_llm(prompt: str, max_tokens: int = 1800) -> str:
    if GROQ_API_KEY:
        headers = {**BROWSER_HEADERS, "Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                model_name = await get_working_groq_model(client, headers)
                res = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.65,
                        "max_tokens": max_tokens
                    }
                )
                if res.status_code == 200:
                    content = res.json()["choices"][0]["message"]["content"]
                    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            except Exception as e:
                print(f"Fallback LLM: {e}")

    async with httpx.AsyncClient(timeout=60.0, headers=BROWSER_HEADERS) as client:
        res = await client.post(
            "https://text.pollinations.ai/openai",
            json={"model": "openai", "messages": [{"role": "user", "content": prompt}]}
        )
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

# === GENERADORES DE GUIONES (VIDEO LARGO + SHORT LANZADOR) ===
async def generate_long_plan(topic: str) -> dict:
    script_prompt = (
        f"Escribe un guion documental fascinante y profundo en español neutro de unos 350 a 450 palabras (3 minutos) sobre: '{topic}'. "
        "Estructura: Gancho inicial impactante, desarrollo con datos sorprendentes divididos en varios puntos y conclusión reflexiva. "
        "REGLAS ESTRICTAS: Devuelve ÚNICAMENTE el texto que dirá el narrador. Sin títulos, sin asteriscos, sin emojis y sin acotaciones."
    )
    script = re.sub(r"[\*\#\[\]\(\)]", "", await ask_llm(script_prompt, max_tokens=2200)).strip()

    title = (await ask_llm(f"Crea un título magnético de YouTube (máximo 65 caracteres) en español para un documental sobre '{topic}'. Devuelve SOLO el título sin comillas.")).replace('"', '').splitlines()[0].strip()[:75]

    terms_raw = await ask_llm(
        f"Based on this documentary topic '{topic}', provide exactly 8 visual search terms in English separated by commas. "
        "Return ONLY the 8 comma-separated terms, nothing else."
    )
    terms = [t.strip() for t in terms_raw.split(",") if t.strip()][:8]
    while len(terms) < 8:
        terms.append("cinematic mystery documentary background")

    return {
        "title": title,
        "description": f"{title}\n\nEn este video exploramos a fondo: {topic}.\n\n{script[:250]}...\n\n#Documental #Misterios #Curiosidades",
        "tags": ["documental", "curiosidades", "misterios", "historia", "ciencia"],
        "script": script,
        "search_terms": terms
    }

async def generate_teaser_short_plan(topic: str, long_title: str, long_url: str) -> dict:
    script_prompt = (
        f"Escribe un guion de YouTube Short de 30 segundos (unas 75 palabras) sobre '{topic}' que funcione como tráiler o gancho lanzador hacia nuestro video largo titulado '{long_title}'. "
        "Empieza con un dato impactante, genera máxima intriga dejando el misterio principal sin resolver y termina diciendo exactamente: "
        "'Acabo de subir el documental completo revelando toda la verdad, toca el enlace de abajo en la pantalla o en el primer comentario para verlo ahora mismo.' "
        "REGLAS: Devuelve ÚNICAMENTE el texto hablado en español neutro, sin asteriscos ni emojis."
    )
    script = re.sub(r"[\*\#\[\]\(\)]", "", await ask_llm(script_prompt, max_tokens=800)).strip()

    title = (await ask_llm(f"Crea un título corto y misterioso (máximo 55 caracteres) para un YouTube Short sobre '{topic}'. SOLO el título sin comillas.")).replace('"', '').splitlines()[0].strip()[:65]

    terms_raw = await ask_llm(
        f"Provide 5 dramatic visual search terms in English separated by commas about '{topic}'. Return ONLY the 5 comma-separated items."
    )
    terms = [t.strip() for t in terms_raw.split(",") if t.strip()][:5]
    while len(terms) < 5:
        terms.append("cinematic dark mystery")

    desc = (
        f"🔥 MIRA EL VIDEO COMPLETO AQUÍ 👉 {long_url}\n\n"
        f"{title}\n\nVideo completo: {long_title} ({long_url})\n\n#Shorts #Curiosidades #Misterios"
    )
    return {
        "title": title,
        "description": desc,
        "tags": ["shorts", "curiosidades", "misterios", "datos"],
        "script": script,
        "search_terms": terms
    }

# === MOTOR AUDIO Y SUBTÍTULOS (.ASS ADAPTATIVO 16:9 Y 9:16) ===
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

def is_valid_image(file_path: str) -> bool:
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 5000:
        return False
    with open(file_path, "rb") as f:
        header = f.read(12)
    return header.startswith(b"\xff\xd8") or header.startswith(b"\x89PNG") or (header.startswith(b"RIFF") and b"WEBP" in header)

async def download_valid_image(client: httpx.AsyncClient, term: str, idx: int, img_file: str, w: int, h: int) -> bool:
    clean_term = re.sub(r"[^a-zA-Z0-9 ]", "", term).strip()
    ratio_str = "vertical 9:16" if h > w else "widescreen 16:9"
    encoded = urllib.parse.quote(f"{clean_term}, {ratio_str}, cinematic lighting, 8k")
    keywords = ",".join(clean_term.split()[:3]) or "space,cinematic"

    urls = [
        f"https://image.pollinations.ai/prompt/{encoded}?width={w}&height={h}&nologo=true&model=flux&seed={idx+300}",
        f"https://loremflickr.com/{w}/{h}/{urllib.parse.quote(keywords)}?lock={idx+20}",
        f"https://picsum.photos/seed/{idx+700}/{w}/{h}"
    ]
    for url in urls:
        try:
            resp = await client.get(url, timeout=25.0)
            if resp.status_code == 200:
                with open(img_file, "wb") as f:
                    f.write(resp.content)
                if is_valid_image(img_file):
                    return True
        except Exception:
            pass
    return False

async def render_video(prefix: str, script_text: str, search_terms: list[str], is_short: bool = True, cta_banner: bool = False) -> str:
    os.makedirs("output", exist_ok=True)
    w, h = (1080, 1920) if is_short else (1920, 1080)
    orientation = "portrait" if is_short else "landscape"

    audio_path = f"output/{prefix}_voice.mp3"
    ass_path = f"output/{prefix}_subs.ass"
    final_video = f"output/{prefix}_final.mp4"

    communicate = edge_tts.Communicate(script_text, voice="es-AR-TomasNeural", rate="+8%" if is_short else "+4%")
    await communicate.save(audio_path)
    total_duration = get_audio_duration(audio_path)

    words = script_text.split()
    time_per_word = total_duration / max(len(words), 1)
    font_size = 95 if is_short else 68
    margin_v = 450 if is_short else 90

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Viral,Liberation Sans,{font_size},&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,3,2,40,40,{margin_v},1
Style: Banner,Liberation Sans,46,&H00FFFFFF,&H000000FF,&H000000FF,&H80000000,-1,0,0,0,100,100,0,0,3,14,0,2,40,40,220,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header)
        if cta_banner and is_short:
            end_all = format_ass_time(total_duration)
            f.write(f"Dialogue: 1,0:00:00.00,{end_all},Banner,,0,0,0,,▼ VIDEO COMPLETO EN EL ENLACE DE ABAJO ▼\n")

        if is_short:
            # 1 palabra por golpe para Shorts
            for idx, word in enumerate(words):
                start = format_ass_time(idx * time_per_word)
                end = format_ass_time((idx + 1) * time_per_word)
                clean = word.strip().upper().replace("{", "").replace("}", "")
                if clean:
                    f.write(f"Dialogue: 0,{start},{end},Viral,,0,0,0,,{clean}\n")
        else:
            # Frases de 4 palabras para el Video Largo horizontal
            chunk_size = 4
            for idx in range(0, len(words), chunk_size):
                phrase = " ".join(words[idx:idx+chunk_size]).upper().replace("{", "").replace("}", "")
                start = format_ass_time(idx * time_per_word)
                end = format_ass_time(min((idx + chunk_size) * time_per_word, total_duration))
                f.write(f"Dialogue: 0,{start},{end},Viral,,0,0,0,,{phrase}\n")

    clip_paths = []
    clip_duration = max(total_duration / len(search_terms), 3.0)

    async with httpx.AsyncClient(timeout=45.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
        for i, term in enumerate(search_terms):
            clip_file = f"output/{prefix}_clip_{i}.mp4"
            got_clip = False

            if PEXELS_API_KEY:
                try:
                    pexels_url = f"https://api.pexels.com/videos/search?query={urllib.parse.quote(term)}&orientation={orientation}&size=medium&per_page=3"
                    r = await client.get(pexels_url, headers={"Authorization": PEXELS_API_KEY})
                    videos = r.json().get("videos", [])
                    if videos:
                        vfiles = videos[0].get("video_files", [])
                        best_file = vfiles[0]
                        vid_data = await client.get(best_file["link"])
                        raw_vid = f"output/{prefix}_raw_{i}.mp4"
                        with open(raw_vid, "wb") as f:
                            f.write(vid_data.content)
                        subprocess.run([
                            "ffmpeg", "-y", "-stream_loop", "-1", "-t", str(clip_duration), "-i", raw_vid,
                            "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps=30",
                            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", clip_file
                        ], check=True)
                        got_clip = True
                except Exception as e:
                    print(f"Pexels fallback ({prefix} {i}): {e}")

            if not got_clip:
                img_file = f"output/{prefix}_img_{i}.jpg"
                has_img = await download_valid_image(client, term, i, img_file, w, h)
                frames = int(clip_duration * 30)
                if has_img:
                    scale_w, scale_h = int(w * 1.15), int(h * 1.15)
                    subprocess.run([
                        "ffmpeg", "-y", "-loop", "1", "-t", str(clip_duration), "-i", img_file,
                        "-vf", f"scale={scale_w}:{scale_h}:force_original_aspect_ratio=increase,crop={scale_w}:{scale_h},zoompan=z='min(zoom+0.0012,1.20)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps=30",
                        "-t", str(clip_duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", clip_file
                    ], check=True)
                else:
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=0x111827:s={w}x{h}:d={clip_duration}:r=30",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", clip_file
                    ], check=True)

            clip_paths.append(str(os.path.abspath(clip_file)))

    concat_file = f"output/{prefix}_concat.txt"
    with open(concat_file, "w") as f:
        for cp in clip_paths:
            f.write(f"file '{cp}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
        "-i", audio_path,
        "-vf", f"ass={ass_path}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k", "-shortest", final_video
    ], check=True)

    return final_video

def upload_youtube_rotated(video_path: str, title: str, description: str, tags: list[str], is_short: bool = True) -> str:
    token_files = sorted(glob.glob("tokens/token_*.json"))
    if not token_files:
        raise RuntimeError("No se encontraron archivos token_*.json.")

    for token_path in token_files:
        try:
            creds = Credentials.from_authorized_user_file(token_path, ["https://www.googleapis.com/auth/youtube.upload"])
            youtube = build("youtube", "v3", credentials=creds)
            body = {
                "snippet": {
                    "title": title[:100],
                    "description": description,
                    "tags": tags,
                    "categoryId": "27"
                },
                "status": {
                    "privacyStatus": "public",
                    "selfDeclaredMadeForKids": False
                }
            }
            media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
            response = youtube.videos().insert(part="snippet,status", body=body, media_body=media).execute()
            vid_id = response["id"]
            return f"https://youtube.com/shorts/{vid_id}" if is_short else f"https://youtu.be/{vid_id}"
        except HttpError as e:
            if e.resp.status in [403, 429] and ("quotaExceeded" in str(e) or "uploadLimitExceeded" in str(e)):
                continue
            raise e
    raise RuntimeError("Todos los tokens agotaron su cuota diaria.")

async def main():
    try:
        mode = "both"
        topic = RAW_TOPIC
        if topic.lower().startswith("/short "):
            mode = "short"
            topic = topic[7:].strip()
        elif topic.lower().startswith("/largo "):
            mode = "long"
            topic = topic[7:].strip()

        long_url = "https://youtube.com"
        long_title = topic

        # 1. PRODUCIR Y SUBIR EL VIDEO LARGO HORIZONTAL (16:9)
        if mode in ["both", "long"]:
            await send_telegram(f"🎥 [1/2] Creando documental largo (16:9) sobre: '{topic}'...")
            long_plan = await generate_long_plan(topic)
            long_title = long_plan["title"]

            await send_telegram(f"🎬 Renderizando Video Largo HD: {long_title}...")
            long_video_path = await render_video("long", long_plan["script"], long_plan["search_terms"], is_short=False)

            await send_telegram("🚀 Subiendo Video Largo a YouTube...")
            long_url = upload_youtube_rotated(long_video_path, long_title, long_plan["description"], long_plan["tags"], is_short=False)
            await send_telegram(f"✅ ¡Video Largo Publicado!\n📺 {long_title}\n🔗 {long_url}")

        # 2. PRODUCIR Y SUBIR EL SHORT LANZADOR VERTICAL (9:16)
        if mode in ["both", "short"]:
            await send_telegram(f"⚡ [2/2] Creando Short Lanzador (9:16) conectado al video largo...")
            short_plan = await generate_teaser_short_plan(topic, long_title, long_url)

            await send_telegram(f"🎬 Renderizando Short Lanzador: {short_plan['title']}...")
            short_video_path = await render_video("short", short_plan["script"], short_plan["search_terms"], is_short=True, cta_banner=(mode == "both"))

            await send_telegram("🚀 Subiendo Short Lanzador a YouTube...")
            short_url = upload_youtube_rotated(short_video_path, short_plan["title"], short_plan["description"], short_plan["tags"], is_short=True)
            await send_telegram(f"🔥 ¡Embudo Completo Publicado!\n\n📺 Video Largo: {long_url}\n📱 Short Lanzador: {short_url}")

    except Exception as e:
        await send_telegram(f"❌ Error en la fábrica: {str(e)}")
        raise e

if __name__ == "__main__":
    asyncio.run(main())     
