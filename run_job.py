import os
import re
import glob
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

# === CONFIGURACIÓN Y VARIABLES DE ENTORNO ===
BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
PEXELS_API_KEY = (os.getenv("PEXELS_API_KEY") or "").strip()

RAW_TOPIC = (os.getenv("TOPIC") or "Las 3 paradojas más extrañas del tiempo").strip()
VOICE_STYLE = (os.getenv("VOICE_STYLE") or "1").strip()
RAW_COUNT = (os.getenv("VIDEO_COUNT") or "1").strip()
MODE_CHOICE = (os.getenv("MODE") or "3").strip()
CHAT_ID = (os.getenv("CHAT_ID") or "").strip()

try:
    VIDEO_COUNT = max(1, min(int(re.sub(r"[^0-9]", "", RAW_COUNT) or "1"), 10))
except ValueError:
    VIDEO_COUNT = 1

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}

ACTIVE_GROQ_MODEL = None

# Inicializar Pipeline de Kokoro-82M en Español Nativo ('e')
print("Inicializando motor neuronal Kokoro-82M (Español 'e')...")
kokoro_pipeline = KPipeline(lang_code='e')

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

# === MOTOR LLM CON AUTO-DETECCIÓN Y FALLBACK ROBUSTO ===
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
    # 1. Intentar con Groq API
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
                print(f"Aviso Groq: {e}")

    # 2. Intentar con Gemini API
    if GEMINI_API_KEY:
        async with httpx.AsyncClient(timeout=60.0, headers=BROWSER_HEADERS) as client:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
                res = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
                if res.status_code == 200:
                    content = res.json()["candidates"][0]["content"]["parts"][0]["text"]
                    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            except Exception as e:
                print(f"Aviso Gemini: {e}")

    # 3. Respaldo Pollinations ajustado (evita errores 400 Bad Request)
    async with httpx.AsyncClient(timeout=60.0, headers=BROWSER_HEADERS) as client:
        try:
            res = await client.post(
                "https://text.pollinations.ai/",
                json={"messages": [{"role": "user", "content": prompt}], "seed": 42}
            )
            if res.status_code == 200:
                if "application/json" in res.headers.get("content-type", ""):
                    content = res.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    content = res.text
                if content.strip():
                    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        except Exception as e:
            print(f"Aviso Pollinations: {e}")

    raise RuntimeError("Todas las APIs de LLM (Groq, Gemini y Pollinations) fallaron al generar el texto.")

# === GENERADOR DE SUBTEMAS PARA LOTES (VIDEO_COUNT > 1) ===
async def generate_subtopics(main_topic: str, count: int) -> list[str]:
    if count <= 1:
        return [main_topic]
    prompt = (
        f"Genera exactamente {count} ángulos o subtemas específicos, intrigantes y diferentes entre sí basados en el tema: '{main_topic}'. "
        f"Devuelve ÚNICAMENTE una lista de {count} líneas (un subtema por línea), sin numeración, sin viñetas ni texto extra."
    )
    raw = await ask_llm(prompt, max_tokens=600)
    lines = [re.sub(r"^[\d\-\.\*\)]+\s*", "", line).strip() for line in raw.splitlines() if line.strip()]
    subtopics = lines[:count]
    while len(subtopics) < count:
        subtopics.append(f"{main_topic} (Parte {len(subtopics) + 1})")
    return subtopics

# === GENERADORES DE GUIONES (LARGO Y SHORT) ===
async def generate_long_plan(topic: str) -> dict:
    script_prompt = (
        f"Escribe un guion documental fascinante y profundo en español neutro de unas 350 a 450 palabras sobre: '{topic}'. "
        "Estructura: Gancho inicial impactante, desarrollo con datos sorprendentes y conclusión reflexiva. "
        "REGLAS ESTRICTAS: Devuelve ÚNICAMENTE el texto que dirá el narrador. Sin títulos, sin asteriscos, sin emojis y sin acotaciones."
    )
    script = re.sub(r"[\*\#\[\]\(\)]", "", await ask_llm(script_prompt, max_tokens=2200)).strip()
    title = (await ask_llm(f"Crea un título magnético de YouTube (máximo 65 caracteres) en español para un documental sobre '{topic}'. SOLO el título sin comillas.")).replace('"', '').splitlines()[0].strip()[:75]

    terms_raw = await ask_llm(
        f"Based on this documentary topic '{topic}', provide exactly 8 visual search terms in English separated by commas. "
        "Return ONLY the 8 comma-separated terms, nothing else."
    )
    terms = [t.strip() for t in terms_raw.split(",") if t.strip()][:8]
    while len(terms) < 8:
        terms.append("cinematic mystery documentary background")

    return {
        "title": title,
        "description": f"{title}\n\nEn este documental exploramos a fondo: {topic}.\n\n{script[:250]}...\n\n#Documental #Misterios #Curiosidades",
        "tags": ["documental", "curiosidades", "misterios", "historia", "ciencia"],
        "script": script,
        "search_terms": terms
    }

async def generate_teaser_short_plan(topic: str, long_title: str, long_url: str, is_standalone: bool = False) -> dict:
    if is_standalone:
        script_prompt = (
            f"Escribe un guion de YouTube Short viral de 35 segundos (unas 85 palabras) sobre '{topic}'. "
            "Empieza con un gancho impactante en los primeros 3 segundos, desarrolla 3 datos sorprendentes y cierra con una pregunta para los comentarios. "
            "REGLAS: Devuelve ÚNICAMENTE el texto hablado en español neutro, sin asteriscos ni emojis."
        )
    else:
        script_prompt = (
            f"Escribe un guion de YouTube Short de 30 segundos (unas 75 palabras) sobre '{topic}' que funcione como gancho lanzador hacia nuestro video largo titulado '{long_title}'. "
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

    if is_standalone:
        desc = f"{title}\n\n{script[:150]}...\n\n#Shorts #Curiosidades #Misterios"
    else:
        desc = (
            f"🔥 MIRA EL DOCUMENTAL COMPLETO AQUÍ 👉 {long_url}\n\n"
            f"{title}\n\nVideo completo: {long_title} ({long_url})\n\n#Shorts #Curiosidades #Misterios"
        )

    return {
        "title": title,
        "description": desc,
        "tags": ["shorts", "curiosidades", "misterios", "datos"],
        "script": script,
        "search_terms": terms
    }

# === MOTOR KOKORO-82M EN ESPAÑOL NATIVO ===
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
    voice_map = {
        "1": "em_alex",
        "2": "ef_dora",
        "3": "em_santa"
    }
    voice_name = voice_map.get(VOICE_STYLE, "em_alex")
    speed_val = 1.08 if is_short else 1.02
    print(f"Generando voz con Kokoro-82M -> Voz: {voice_name} | Velocidad: {speed_val}x")

    generator = kokoro_pipeline(script_text, voice=voice_name, speed=speed_val)
    audio_chunks = [chunk for _, _, chunk in generator]

    if audio_chunks:
        complete_audio = np.concatenate(audio_chunks)
        sf.write(audio_path, complete_audio, 24000)
    else:
        raise RuntimeError("Kokoro-82M no devolvió audio para el guion.")

# === DESCARGA VISUAL VALIDADA Y ENSAMBLAJE FFMPEG ===
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

    audio_path = f"output/{prefix}_voice.wav"
    ass_path = f"output/{prefix}_subs.ass"
    final_video = f"output/{prefix}_final.mp4"

    await generate_voice_kokoro(script_text, audio_path, is_short=is_short)
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
            for idx, word in enumerate(words):
                start = format_ass_time(idx * time_per_word)
                end = format_ass_time((idx + 1) * time_per_word)
                clean = word.strip().upper().replace("{", "").replace("}", "")
                if clean:
                    f.write(f"Dialogue: 0,{start},{end},Viral,,0,0,0,,{clean}\n")
        else:
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

# === SUBIDA A YOUTUBE CON ROTADOR DE CREDENCIALES ===
def upload_youtube_rotated(video_path: str, title: str, description: str, tags: list[str], is_short: bool = True) -> str:
    token_files = sorted(glob.glob("tokens/token_*.json"))
    if not token_files:
        raise RuntimeError("No se encontraron archivos token_*.json en /tokens.")

    for token_path in token_files:
        try:
            creds = Credentials.from_authorized_user_file(token_path, ["https://www.googleapis.com/auth/youtube.upload"])
            youtube = build("youtube", "v3", credentials=creds)
            body = {
                "snippet": {"title": title[:100], "description": description, "tags": tags, "categoryId": "27"},
                "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False}
            }
            media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")
            response = youtube.videos().insert(part="snippet,status", body=body, media_body=media).execute()
            vid_id = response["id"]
            return f"https://youtube.com/shorts/{vid_id}" if is_short else f"https://youtu.be/{vid_id}"
        except HttpError as e:
            if e.resp.status in [403, 429] and ("quotaExceeded" in str(e) or "uploadLimitExceeded" in str(e)):
                print(f"Cuota agotada en {token_path}, rotando al siguiente token...")
                continue
            raise e
    raise RuntimeError("Todos los tokens de YouTube agotaron su cuota diaria.")

# === ORQUESTADOR PRINCIPAL CON BUCLE DE CANTIDAD (VIDEO_COUNT) ===
async def main():
    try:
        mode = "both"
        if MODE_CHOICE == "1":
            mode = "short"
        elif MODE_CHOICE == "2":
            mode = "long"
        elif MODE_CHOICE == "3":
            mode = "both"

        voice_labels = {"1": "em_alex (Masculina Neutra)", "2": "ef_dora (Femenina Dinámica)", "3": "em_santa (Documental Profunda)"}
        chosen_voice_label = voice_labels.get(VOICE_STYLE, "em_alex")

        await send_telegram(
            f"⚙️ Iniciando tanda de producción:\n"
            f"📌 Tema base: {RAW_TOPIC}\n"
            f"🎙️ Voz Kokoro: {chosen_voice_label}\n"
            f"📦 Cantidad: {VIDEO_COUNT}\n"
            f"🎬 Modo: {mode.upper()}"
        )

        subtopics = await generate_subtopics(RAW_TOPIC, VIDEO_COUNT)

        for idx, current_topic in enumerate(subtopics, start=1):
            prefix_tag = f"[{idx}/{VIDEO_COUNT}]"
            long_url = "https://youtube.com"
            long_title = current_topic

            # 1. VIDEO LARGO HORIZONTAL (16:9)
            if mode in ["both", "long"]:
                await send_telegram(f"🎥 {prefix_tag} Creando documental largo (16:9) sobre: '{current_topic}'...")
                long_plan = await generate_long_plan(current_topic)
                long_title = long_plan["title"]

                await send_telegram(f"🎬 {prefix_tag} Renderizando Video Largo: {long_title}...")
                long_video_path = await render_video(f"long_{idx}", long_plan["script"], long_plan["search_terms"], is_short=False)

                await send_telegram(f"🚀 {prefix_tag} Subiendo Video Largo a YouTube...")
                long_url = upload_youtube_rotated(long_video_path, long_title, long_plan["description"], long_plan["tags"], is_short=False)
                await send_telegram(f"✅ {prefix_tag} ¡Video Largo Publicado!\n📺 {long_title}\n🔗 {long_url}")

            # 2. SHORT VERTICAL (9:16)
            if mode in ["both", "short"]:
                is_standalone = (mode == "short")
                await send_telegram(f"⚡ {prefix_tag} Creando Short (9:16) sobre: '{current_topic}'...")
                short_plan = await generate_teaser_short_plan(current_topic, long_title, long_url, is_standalone=is_standalone)

                await send_telegram(f"🎬 {prefix_tag} Renderizando Short: {short_plan['title']}...")
                short_video_path = await render_video(
                    f"short_{idx}",
                    short_plan["script"],
                    short_plan["search_terms"],
                    is_short=True,
                    cta_banner=(mode == "both")
                )

                await send_telegram(f"🚀 {prefix_tag} Subiendo Short a YouTube...")
                short_url = upload_youtube_rotated(short_video_path, short_plan["title"], short_plan["description"], short_plan["tags"], is_short=True)

                if mode == "both":
                    await send_telegram(f"🔥 {prefix_tag} ¡Embudo Completado!\n📺 Largo: {long_url}\n📱 Short: {short_url}")
                else:
                    await send_telegram(f"✅ {prefix_tag} ¡Short Publicado!\n📌 {short_plan['title']}\n🔗 {short_url}")

        await send_telegram(f"🏁 ¡Tanda de {VIDEO_COUNT} producción(es) finalizada con éxito!")

    except Exception as e:
        await send_telegram(f"❌ Error en la fábrica: {str(e)}")
        raise e

if __name__ == "__main__":
    asyncio.run(main())
