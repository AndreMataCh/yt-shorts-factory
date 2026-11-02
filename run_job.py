import os
import json
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
TOPIC = os.getenv("TOPIC") or "Las 3 paradojas más extrañas del tiempo"
CHAT_ID = (os.getenv("CHAT_ID") or "").strip()

async def send_telegram(text: str):
    print(text)
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={"chat_id": int(CHAT_ID), "text": text})

def format_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

async def get_active_groq_model(client: httpx.AsyncClient, headers: dict) -> str:
    """Detecta automáticamente el mejor modelo activo hoy en Groq para evitar errores 404."""
    res = await client.get("https://api.groq.com/openai/v1/models", headers=headers)
    if res.status_code != 200:
        raise RuntimeError(f"Error autenticando en Groq ({res.status_code}): {res.text}")
    
    available = [m["id"] for m in res.json().get("data", []) if "whisper" not in m["id"] and "tts" not in m["id"] and "guard" not in m["id"]]
    print(f"Modelos activos detectados en Groq: {available}")

    # Orden de preferencia
    preferred = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "qwen-qwq-32b",
        "gemma2-9b-it"
    ]
    for p in preferred:
        if p in available:
            return p
    # Si todos cambiaron de nombre, elige el primer modelo Llama o general disponible
    for m in available:
        if "llama" in m.lower():
            return m
    return available[0]

import re

async def generate_script(topic: str) -> dict:
    prompt = f"""Crea un YouTube Short viral de 35 segundos sobre: "{topic}".
Responde UNICAMENTE con un objeto JSON válido (sin texto adicional ni bloques markdown), asegurándote de cerrar todas las comillas y llaves:
{{
  "title": "Título magnético con gancho (máx 70 caracteres)",
  "description": "Descripción SEO atractiva con hashtags",
  "tags": ["curiosidades", "datos", "shorts"],
  "script": "Guion narrado en español neutro en un solo párrafo continuo sin saltos de línea ni emojis, directo para leer en voz alta.",
  "image_prompts": [
    "Detailed vertical 9:16 visual prompt in English for scene 1",
    "Detailed vertical 9:16 visual prompt in English for scene 2",
    "Detailed vertical 9:16 visual prompt in English for scene 3",
    "Detailed vertical 9:16 visual prompt in English for scene 4",
    "Detailed vertical 9:16 visual prompt in English for scene 5"
  ]
}}"""

    async with httpx.AsyncClient(timeout=45.0, headers=BROWSER_HEADERS) as client:
        if GROQ_API_KEY:
            print("Generando guion con Groq API...")
            headers = {
                **BROWSER_HEADERS,
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            res = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": "You are a JSON generator. Always output valid, complete JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.5,
                    "max_tokens": 1500
                }
            )
            if res.status_code != 200:
                raise RuntimeError(f"Error en Groq API ({res.status_code}): {res.text}")
            
            raw_text = res.json()["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if not match:
                raise RuntimeError(f"No se encontró JSON en la respuesta: {raw_text}")
            
            clean_json = match.group(0)
            return json.loads(clean_json, strict=False)

        raise RuntimeError("No se encontró GROQ_API_KEY en Repository secrets.")

async def transcribe_groq_whisper(audio_path: str) -> tuple[list, float]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(audio_path, "rb") as f:
            res = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": ("voice.mp3", f, "audio/mpeg")},
                data={
                    "model": "whisper-large-v3-turbo",
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "word"
                }
            )
            if res.status_code != 200:
                raise RuntimeError(f"Groq Whisper Error ({res.status_code}): {res.text}")
            data = res.json()
            return data.get("words", []), float(data.get("duration", 35.0))

async def build_video(script_text: str, image_prompts: list[str]) -> str:
    os.makedirs("output", exist_ok=True)
    audio_path = "output/voice.mp3"
    ass_path = "output/subs.ass"
    final_video = "output/final_short.mp4"

    # 1. Voz Neuronal Gratis (Edge-TTS)
    communicate = edge_tts.Communicate(script_text, voice="es-ES-AlvaroNeural", rate="+8%")
    await communicate.save(audio_path)

    # 2. Subtítulos palabra por palabra vía Groq Whisper API
    words, total_duration = await transcribe_groq_whisper(audio_path)

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
        for w in words:
            start = format_ass_time(float(w["start"]))
            end = format_ass_time(float(w["end"]))
            clean_word = w["word"].strip().upper().replace("{", "").replace("}", "")
            if clean_word:
                f.write(f"Dialogue: 0,{start},{end},Viral,,0,0,0,,{clean_word}\n")

    # 3. Descargar imágenes con Flux (Pollinations) y aplicar Zoom con FFmpeg
    clip_paths = []
    clip_duration = max(total_duration / len(image_prompts), 2.5)

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        for i, prompt in enumerate(image_prompts):
            encoded = urllib.parse.quote(f"{prompt}, vertical 9:16, cinematic lighting, 8k, hyperrealistic")
            url = f"https://image.pollinations.ai/prompt/{encoded}?width=1080&height=1920&nologo=true&seed={i+100}"
            img_file = f"output/img_{i}.jpg"
            resp = await client.get(url)
            with open(img_file, "wb") as f:
                f.write(resp.content)

            clip_file = f"output/clip_{i}.mp4"
            frames = int(clip_duration * 30)
            subprocess.run([
                "ffmpeg", "-y", "-loop", "1", "-i", img_file,
                "-vf", f"scale=1200:-1,zoompan=z='min(zoom+0.0015,1.25)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s=1080x1920:fps=30",
                "-t", str(clip_duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", clip_file
            ], check=True)
            clip_paths.append(str(os.path.abspath(clip_file)))

    # 4. Unir escenas y quemar subtítulos + audio
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

def upload_youtube_rotated(video_path: str, title: str, description: str, tags: list[str]) -> str:
    token_files = sorted(glob.glob("tokens/token_*.json"))
    if not token_files:
        raise RuntimeError("No se encontraron archivos token_*.json en /tokens")

    for token_path in token_files:
        try:
            creds = Credentials.from_authorized_user_file(token_path, ["https://www.googleapis.com/auth/youtube.upload"])
            youtube = build("youtube", "v3", credentials=creds)
            body = {
                "snippet": {
                    "title": title[:100],
                    "description": f"{description}\n\n#Shorts",
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
        await send_telegram(f"🧠 Creando guion y escenas para: '{TOPIC}'...")
        plan = await generate_script(TOPIC)

        await send_telegram(f"🎬 Renderizando Short: {plan['title']}...")
        video_path = await build_video(plan["script"], plan["image_prompts"])

        await send_telegram("🚀 Subiendo a YouTube con rotador OAuth...")
        yt_url = upload_youtube_rotated(video_path, plan["title"], plan["description"], plan["tags"])

        await send_telegram(f"✅ ¡Short publicado con éxito!\n📌 {plan['title']}\n🔗 {yt_url}")
    except Exception as e:
        await send_telegram(f"❌ Error en la fábrica: {str(e)}")
        raise e

if __name__ == "__main__":
    asyncio.run(main())
