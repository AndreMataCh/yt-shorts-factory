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

# === CONFIGURACIÓN Y VARIABLES DE ENTORNO ===
BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
PEXELS_API_KEY = (os.getenv("PEXELS_API_KEY") or "").strip()
TOPIC = (os.getenv("TOPIC") or "Las 3 paradojas más extrañas del tiempo").strip()
CHAT_ID = (os.getenv("CHAT_ID") or "").strip()

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

# === AUTO-DETECTOR DE MODELOS ACTIVOS EN GROQ + RESPALDO GRATUITO ===
async def get_working_groq_model(client: httpx.AsyncClient, headers: dict) -> str:
    global ACTIVE_GROQ_MODEL
    if ACTIVE_GROQ_MODEL:
        return ACTIVE_GROQ_MODEL

    res = await client.get("https://api.groq.com/openai/v1/models", headers=headers)
    if res.status_code != 200:
        raise RuntimeError(f"Error consultando modelos en Groq ({res.status_code}): {res.text}")

    all_models = [m["id"] for m in res.json().get("data", [])]
    # Filtrar modelos de audio, visión pura o moderación (guard)
    chat_models = [
        m for m in all_models
        if not any(x in m.lower() for x in ["whisper", "guard", "tts", "playai", "vision", "embed", "orpheus"])
    ]
    print(f"Modelos de texto disponibles en tu cuenta Groq: {chat_models}")

    # Priorizar modelos de mayor calidad (70b, llama-4, qwen, deepseek, 8b)
    for keyword in ["70b", "llama-4", "qwen", "deepseek", "llama-3", "llama3", "gemma", "mixtral", "8b"]:
        for m in chat_models:
            if keyword in m.lower():
                ACTIVE_GROQ_MODEL = m
                print(f"--> Modelo seleccionado automáticamente: {ACTIVE_GROQ_MODEL}")
                return ACTIVE_GROQ_MODEL

    if chat_models:
        ACTIVE_GROQ_MODEL = chat_models[0]
        return ACTIVE_GROQ_MODEL

    raise RuntimeError("No se encontraron modelos de chat activos en esta cuenta de Groq.")

async def ask_llm(prompt: str, system: str = "You are a helpful assistant.") -> str:
    # Intento 1: Groq con detección dinámica del modelo activo
    if GROQ_API_KEY:
        headers = {
            **BROWSER_HEADERS,
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient(timeout=45.0) as client:
            try:
                model_name = await get_working_groq_model(client, headers)
                res = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.6,
                        "max_tokens": 1000
                    }
                )
                if res.status_code == 200:
                    content = res.json()["choices"][0]["message"]["content"]
                    # Limpiar bloques <think>...</think> por si selecciona un modelo razonador (DeepSeek/Qwen)
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                    return content
                print(f"Aviso Groq ({res.status_code}): {res.text}. Usando motor de respaldo...")
            except Exception as e:
                print(f"Aviso en Groq: {e}. Saltando al motor gratuito de respaldo...")

    # Intento 2 (Respaldo infalible sin API Key): Pollinations OpenAI
    async with httpx.AsyncClient(timeout=60.0, headers=BROWSER_HEADERS) as client:
        res = await client.post(
            "https://text.pollinations.ai/openai",
            json={
                "model": "openai",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
            }
        )
        res.raise_for_status()
        content = res.json()["choices"][0]["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

# === PASO 1: GENERADOR MODULAR ESTILO MONEYPRINTERTURBO ===
async def generate_mpt_content(topic: str) -> dict:
    script_prompt = (
        f"Escribe un guion para un YouTube Short viral de 35 segundos sobre: '{topic}'. "
        "REGLAS ESTRICTAS: Devuelve ÚNICAMENTE el texto que dirá el narrador en español neutro. "
        "No incluyas títulos, ni etiquetas como 'Narrador:', ni emojis, ni asteriscos, ni acotaciones entre paréntesis. "
        "Escribe todo en un solo párrafo fluido y atrapante."
    )
    script = await ask_llm(script_prompt)
    script = re.sub(r"[\*\#\[\]\(\)]", "", script).strip()

    title_prompt = (
        f"Crea un título viral y curioso de máximo 60 caracteres para un video sobre '{topic}'. "
        "Devuelve SOLO el título en español, sin comillas ni texto extra."
    )
    title = (await ask_llm(title_prompt)).replace('"', '').splitlines()[0].strip()[:70]

    terms_prompt = (
        f"Based on this script: '{script}', provide exactly 5 visual scene descriptions in English "
        "separated by a comma. Return ONLY the 5 comma-separated items, nothing else. "
        "Example: galaxies spinning in space, antique pocket watch, futuristic city night, glowing brain neurons, dark ocean waves"
    )
    raw_terms = await ask_llm(terms_prompt)
    search_terms = [t.strip() for t in raw_terms.split(",") if t.strip()][:5]
    while len(search_terms) < 5:
        search_terms.append("cinematic abstract universe background")

    return {
        "title": title,
        "description": f"{title}\n\n{script[:150]}...\n\n#Shorts #Curiosidades #DatosCuriosos",
        "tags": ["shorts", "curiosidades", "datos", "misterios", "ciencia"],
        "script": script,
        "search_terms": search_terms
    }

# === PASO 2: VOZ Y SUBTÍTULOS NATIVOS (EDGE-TTS) ===
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

# === PASO 3: DESCARGA DE MATERIAL VISUAL (PEXELS VIDEO HD + FALLBACK IA) ===
async def get_visual_clips(search_terms: list[str], total_duration: float) -> list[str]:
    clip_paths = []
    clip_duration = max(total_duration / len(search_terms), 2.5)

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
        for i, term in enumerate(search_terms):
            clip_file = f"output/clip_{i}.mp4"
            got_pexels = False

            if PEXELS_API_KEY:
                try:
                    pexels_url = f"https://api.pexels.com/videos/search?query={urllib.parse.quote(term)}&orientation=portrait&size=medium&per_page=3"
                    r = await client.get(pexels_url, headers={"Authorization": PEXELS_API_KEY})
                    videos = r.json().get("videos", [])
                    if videos:
                        video_files = videos[0].get("video_files", [])
                        best_file = next((vf for vf in video_files if vf.get("width", 0) >= 720 and vf.get("height", 0) > vf.get("width", 0)), video_files[0])
                        vid_data = await client.get(best_file["link"])
                        raw_vid = f"output/raw_{i}.mp4"
                        with open(raw_vid, "wb") as f:
                            f.write(vid_data.content)
                        subprocess.run([
                            "ffmpeg", "-y", "-stream_loop", "-1", "-i", raw_vid,
                            "-t", str(clip_duration),
                            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30",
                            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", clip_file
                        ], check=True)
                        got_pexels = True
                except Exception as e:
                    print(f"Pexels fallback en escena {i}: {e}")

            if not got_pexels:
                encoded = urllib.parse.quote(f"{term}, vertical 9:16, cinematic lighting, 8k, hyperrealistic")
                url = f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&nologo=true&seed={i+105}"
                img_file = f"output/img_{i}.jpg"
                resp = await client.get(url)
                with open(img_file, "wb") as f:
                    f.write(resp.content)

                frames = int(clip_duration * 30)
                subprocess.run([
                    "ffmpeg", "-y", "-loop", "1", "-i", img_file,
                    "-vf", f"scale=1200:-1,zoompan=z='min(zoom+0.0015,1.25)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1080x1920:fps=30",
                    "-t", str(clip_duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", clip_file
                ], check=True)

            clip_paths.append(str(os.path.abspath(clip_file)))
    return clip_paths

async def build_video(script_text: str, search_terms: list[str]) -> str:
    os.makedirs("output", exist_ok=True)
    audio_path = "output/voice.mp3"
    ass_path = "output/subs.ass"
    final_video = "output/final_short.mp4"

    communicate = edge_tts.Communicate(script_text, voice="es-ES-AlvaroNeural", rate="+8%")
    await communicate.save(audio_path)
    total_duration = get_audio_duration(audio_path)
    
    words = script_text.split()
    time_per_word = total_duration / max(len(words), 1)

    ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Viral,Liberation Sans,95,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,3,5,40,40,450,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_header)
        for idx, w in enumerate(words):
            start = format_ass_time(idx * time_per_word)
            end = format_ass_time((idx + 1) * time_per_word)
            clean_word = w.strip().upper().replace("{", "").replace("}", "")
            if clean_word:
                f.write(f"Dialogue: 0,{start},{end},Viral,,0,0,0,,{clean_word}\n")

    clip_paths = await get_visual_clips(search_terms, total_duration)

    concat_file = "output/concat.txt"
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

# === PASO 4: SUBIDA A YOUTUBE CON ROTADOR MULTI-PROYECTO ===
def upload_youtube_rotated(video_path: str, title: str, description: str, tags: list[str]) -> str:
    token_files = sorted(glob.glob("tokens/token_*.json"))
    if not token_files:
        raise RuntimeError("No se encontraron archivos token_*.json. Revisa el secreto YT_TOKEN_1.")

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
            return f"https://youtube.com/shorts/{response['id']}"
        except HttpError as e:
            if e.resp.status in [403, 429] and ("quotaExceeded" in str(e) or "uploadLimitExceeded" in str(e)):
                print(f"Cuota agotada en {token_path}, rotando al siguiente...")
                continue
            raise e
    raise RuntimeError("Todos los tokens agotaron su cuota diaria.")

async def main():
    try:
        await send_telegram(f"🧠 [Motor MoneyPrinter] Creando guion para: '{TOPIC}'...")
        plan = await generate_mpt_content(TOPIC)
        print(f"Título generado: {plan['title']}")
        print(f"Escenas: {plan['search_terms']}")

        await send_telegram(f"🎬 Renderizando Short: {plan['title']}...")
        video_path = await build_video(plan["script"], plan["search_terms"])

        await send_telegram("🚀 Subiendo a YouTube con rotador OAuth...")
        yt_url = upload_youtube_rotated(video_path, plan["title"], plan["description"], plan["tags"])

        await send_telegram(f"✅ ¡Short publicado con éxito!\n📌 {plan['title']}\n🔗 {yt_url}")
    except Exception as e:
        await send_telegram(f"❌ Error en la fábrica: {str(e)}")
        raise e

if __name__ == "__main__":
    asyncio.run(main())                                        
