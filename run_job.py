import os
import re
import glob
import asyncio
import subprocess
import urllib.parse
from datetime import datetime, timedelta, timezone
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

CHANNEL_ID = (os.getenv("CHANNEL_ID") or "ch1").strip().lower()
RAW_TOPIC = (os.getenv("TOPIC") or "Las 3 paradojas más extrañas del tiempo").strip()
VISUAL_STYLE = (os.getenv("VISUAL_STYLE") or "1").strip()  # "1" = Realista, "2" = Animado 3D/Anime
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

CACHED_GROQ_MODELS = []
CACHED_GEMINI_MODELS = []

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

def clean_llm_output(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()

# === AUTODESCUBRIMIENTO DE MODELOS EN GEMINI Y GROQ ===
async def get_ranked_gemini_models(client: httpx.AsyncClient) -> list[str]:
    global CACHED_GEMINI_MODELS
    if CACHED_GEMINI_MODELS:
        return CACHED_GEMINI_MODELS

    default_models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
        res = await client.get(url)
        if res.status_code == 200:
            data = res.json().get("models", [])
            valid = []
            for m in data:
                name = m.get("name", "").replace("models/", "")
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" in methods and not any(
                    x in name.lower() for x in ["vision", "embedding", "imagen", "veo", "tts", "audio", "aqa", "bisheng"]
                ):
                    valid.append(name)

            ranked = []
            for pref in ["gemini-2.5-flash", "gemini-2.5-pro", "2.5-flash", "flash", "pro"]:
                for v in valid:
                    if pref in v.lower() and v not in ranked:
                        ranked.append(v)
            for v in valid:
                if v not in ranked:
                    ranked.append(v)

            if ranked:
                CACHED_GEMINI_MODELS = ranked
                return CACHED_GEMINI_MODELS
    except Exception as e:
        print(f"Aviso listando modelos Gemini: {e}")

    CACHED_GEMINI_MODELS = default_models
    return CACHED_GEMINI_MODELS

async def get_ranked_groq_models(client: httpx.AsyncClient, headers: dict) -> list[str]:
    global CACHED_GROQ_MODELS
    if CACHED_GROQ_MODELS:
        return CACHED_GROQ_MODELS

    res = await client.get("https://api.groq.com/openai/v1/models", headers=headers)
    if res.status_code != 200:
        raise RuntimeError(f"Error Groq Models ({res.status_code}): {res.text}")

    all_models = [m["id"] for m in res.json().get("data", [])]
    chat_models = [
        m for m in all_models
        if not any(x in m.lower() for x in ["whisper", "guard", "tts", "playai", "vision", "embed", "orpheus", "allam"])
    ]

    fast_models = [m for m in chat_models if not any(r in m.lower() for r in ["qwen", "deepseek", "r1"])]
    reasoning_models = [m for m in chat_models if m not in fast_models]

    priority_keywords = ["llama-3.3", "llama-3.1", "llama-4", "llama", "gpt-oss", "kimi", "gemma", "mixtral"]
    ranked = []
    for kw in priority_keywords:
        for m in fast_models:
            if kw in m.lower() and m not in ranked:
                ranked.append(m)
    for m in fast_models:
        if m not in ranked:
            ranked.append(m)
    ranked.extend(reasoning_models)

    CACHED_GROQ_MODELS = ranked
    return CACHED_GROQ_MODELS

async def ask_llm(prompt: str, max_tokens: int = 1200) -> str:
    errors = []

    if GEMINI_API_KEY:
        async with httpx.AsyncClient(timeout=60.0, headers=BROWSER_HEADERS) as client:
            gemini_models = await get_ranked_gemini_models(client)
            for gemini_model in gemini_models[:3]:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={GEMINI_API_KEY}"
                    res = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
                    if res.status_code == 200:
                        candidates = res.json().get("candidates", [])
                        if candidates:
                            content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                            cleaned = clean_llm_output(content)
                            if cleaned:
                                return cleaned
                    elif res.status_code == 429:
                        errors.append(f"Gemini 429 ({gemini_model})")
                        await asyncio.sleep(4)
                    else:
                        errors.append(f"Gemini {res.status_code} ({gemini_model})")
                except Exception as e:
                    errors.append(f"Gemini Exception ({gemini_model}): {e}")

    if GROQ_API_KEY:
        headers = {**BROWSER_HEADERS, "Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                models_to_try = await get_ranked_groq_models(client, headers)
                for model_name in models_to_try[:5]:
                    is_qwen = "qwen" in model_name.lower()
                    effective_prompt = f"/no_think\n{prompt}" if is_qwen else prompt

                    res = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json={
                            "model": model_name,
                            "messages": [{"role": "user", "content": effective_prompt}],
                            "temperature": 0.7,
                            "max_tokens": max_tokens
                        }
                    )
                    if res.status_code == 200:
                        msg_obj = res.json()["choices"][0]["message"]
                        content = msg_obj.get("content") or ""
                        cleaned = clean_llm_output(content)
                        if cleaned:
                            return cleaned
                    elif res.status_code == 429:
                        errors.append(f"Groq 429 ({model_name})")
                        await asyncio.sleep(6)
                        continue
                    else:
                        errors.append(f"Groq {res.status_code} ({model_name})")
            except Exception as e:
                errors.append(f"Groq Exception: {e}")

    async with httpx.AsyncClient(timeout=60.0, headers=BROWSER_HEADERS, follow_redirects=True) as client:
        try:
            encoded_p = urllib.parse.quote(prompt[:1500])
            seed = np.random.randint(1000, 99999)
            res = await client.get(f"https://text.pollinations.ai/{encoded_p}?seed={seed}")
            if res.status_code == 200 and res.text.strip():
                cleaned = clean_llm_output(res.text)
                if cleaned:
                    return cleaned
            else:
                errors.append(f"Pollinations GET {res.status_code}")
        except Exception as e:
            errors.append(f"Pollinations Exception: {e}")

    raise RuntimeError(f"Falla total de LLMs. Detalles: {' | '.join(errors)}")

# === ESTUDIO DE COMPETENCIA Y MODELADO DE ESTRATEGIAS VIRALES ===
async def analyze_niche_competitors(main_topic: str) -> dict:
    await send_telegram(f"🔍 Analizando canales exitosos y patrones virales en el nicho: '{main_topic}'...")

    prompt = (
        f"Actúa como un estratega experto en crecimiento de YouTube y analista de datos de contenidos virales. "
        f"Para el nicho o tema general: '{main_topic}':\n"
        f"1. Identifica las 2 estrategias o ganchos psicológicos principales que utilizan los canales más exitosos de este nicho para retener audiencia.\n"
        f"2. Basándote en esos patrones ganadores, diseña un subtema o enfoque específico altamente magnético y novedoso.\n"
        f"3. Devuelve la respuesta sin razonamientos previos en este formato exacto de texto:\n"
        f"ESTRATEGIA: [Explica brevemente la estrategia de la competencia]\n"
        f"TEMA_OPTIMIZADO: [El título o enfoque exacto perfeccionado para el video]\n"
        f"GANCHO: [La frase exacta con la que debe iniciar el guion para atrapar al usuario en 3 segundos]"
    )

    response = await ask_llm(prompt, max_tokens=600)

    estrategia = "Patrones de alta retención estándar"
    tema_optimizado = main_topic
    gancho = ""

    for line in response.splitlines():
        if line.startswith("ESTRATEGIA:"):
            estrategia = line.replace("ESTRATEGIA:", "").strip()
        elif line.startswith("TEMA_OPTIMIZADO:"):
            tema_optimizado = line.replace("TEMA_OPTIMIZADO:", "").strip()
        elif line.startswith("GANCHO:"):
            gancho = line.replace("GANCHO:", "").strip()

    if not tema_optimizado:
        tema_optimizado = main_topic

    return {"strategy": estrategia, "topic": tema_optimizado, "hook": gancho}

async def generate_subtopics(main_topic: str, count: int) -> list[str]:
    if count <= 1:
        return [main_topic]
    prompt = (
        f"Genera exactamente {count} ángulos, enfoques o subtemas totalmente distintos y creativos basados en: '{main_topic}'. "
        f"Evita repetir conceptos. Devuelve ÚNICAMENTE una lista de {count} líneas (un subtema por línea), sin numeración ni texto extra."
    )
    raw = await ask_llm(prompt, max_tokens=500)
    lines = [re.sub(r"^[\d\-\.\*\)]+\s*", "", line).strip() for line in raw.splitlines() if line.strip()]
    subtopics = lines[:count]
    while len(subtopics) < count:
        subtopics.append(f"{main_topic} - Parte {len(subtopics) + 1}")
    return subtopics

# === GENERADORES DE GUIONES (CON SOPORTE PARA ESCENAS REALISTAS O ANIMADAS) ===
async def generate_long_plan(topic: str) -> dict:
    script_prompt = (
        f"Escribe un guion fascinante y profundo en español neutro de unas 350 a 450 palabras sobre: '{topic}'. "
        "Estructura: Gancho inicial impactante, desarrollo envolvente y conclusión memorable. "
        "REGLAS ESTRICTAS: Devuelve ÚNICAMENTE el texto que dirá el narrador. Sin títulos, sin asteriscos, sin emojis."
    )
    script = re.sub(r"[\*\#\[\]\(\)]", "", await ask_llm(script_prompt, max_tokens=1400)).strip()
    title = (await ask_llm(f"Crea un título magnético de YouTube (máx 65 caracteres) en español sobre '{topic}'. SOLO el título sin comillas.", max_tokens=150)).replace('"', '').splitlines()[0].strip()[:75]

    visual_instruction = (
        "vivid 3D animated movie scenes, expressive characters, magical environments, studio ghibli and pixar style"
        if VISUAL_STYLE == "2" else
        "cinematic close-ups, wide landscape shots, abstract conceptual imagery, dynamic movement"
    )
    terms_raw = await ask_llm(
        f"For the video topic '{topic}', provide exactly 8 completely unique and highly descriptive visual scene prompts in English ({visual_instruction}). "
        "Return ONLY the 8 comma-separated scene descriptions.",
        max_tokens=300
    )
    terms = [t.strip() for t in terms_raw.split(",") if t.strip()][:8]
    while len(terms) < 8:
        terms.append("magical glowing universe scene" if VISUAL_STYLE == "2" else "cinematic macro abstract background")

    return {
        "title": title,
        "description": f"{title}\n\nExploramos a fondo: {topic}.\n\n{script[:250]}...\n\n#Documental #Misterios #Historias",
        "tags": ["documental", "misterios", "ciencia", "historia", "animacion" if VISUAL_STYLE == "2" else "curiosidades"],
        "script": script,
        "search_terms": terms
    }

async def generate_teaser_short_plan(topic: str, long_title: str, long_url: str, is_standalone: bool = False) -> dict:
    if is_standalone:
        script_prompt = (
            f"Escribe un guion de YouTube Short viral de 35 segundos sobre '{topic}'. "
            "Empieza con un gancho potente y cierra con una pregunta. "
            "REGLAS: Devuelve ÚNICAMENTE el texto hablado en español neutro, sin asteriscos ni emojis."
        )
    else:
        script_prompt = (
            f"Escribe un guion de YouTube Short de 30 segundos sobre '{topic}' como gancho hacia nuestro video largo '{long_title}'. "
            "Termina diciendo exactamente: 'Acabo de subir el video completo revelando toda la historia, toca el enlace de abajo en la pantalla o en el primer comentario para verlo ahora mismo.' "
            "REGLAS: Devuelve ÚNICAMENTE el texto hablado en español neutro, sin asteriscos ni emojis."
        )

    script = re.sub(r"[\*\#\[\]\(\)]", "", await ask_llm(script_prompt, max_tokens=600)).strip()
    title = (await ask_llm(f"Crea un título corto y magnético (máx 55 caracteres) para un Short sobre '{topic}'. SOLO el título sin comillas.", max_tokens=150)).replace('"', '').splitlines()[0].strip()[:65]

    visual_instruction = (
        "3D Pixar/Anime animated scenes, vibrant colors, expressive characters"
        if VISUAL_STYLE == "2" else
        "high-impact cinematic visual search terms"
    )
    terms_raw = await ask_llm(
        f"Provide 5 distinct {visual_instruction} in English about '{topic}'. Return ONLY the 5 comma-separated descriptions.",
        max_tokens=200
    )
    terms = [t.strip() for t in terms_raw.split(",") if t.strip()][:5]
    while len(terms) < 5:
        terms.append("vibrant 3d animated fantasy world" if VISUAL_STYLE == "2" else "dark cinematic aesthetic atmosphere")

    desc = f"{title}\n\n🔗 Video completo: {long_url}\n\n#Shorts #Curiosidades" if not is_standalone else f"{title}\n\n#Shorts"

    return {
        "title": title,
        "description": desc,
        "tags": ["shorts", "curiosidades", "animacion" if VISUAL_STYLE == "2" else "misterios"],
        "script": script,
        "search_terms": terms
    }

# === MOTOR KOKORO-82M Y MÚSICA AMBIENTAL AL 12% ===
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
    speed_val = 1.08 if is_short else 1.02

    generator = kokoro_pipeline(script_text, voice=voice_name, speed=speed_val)
    audio_chunks = [chunk for _, _, chunk in generator]

    if audio_chunks:
        complete_audio = np.concatenate(audio_chunks)
        sf.write(audio_path, complete_audio, 24000)
    else:
        raise RuntimeError("Kokoro-82M no devolvió audio.")

async def prepare_background_music(duration: float, bgm_path: str):
    """
    Descarga una pista instrumental libre de derechos o genera una atmósfera
    cinematográfica envolvente en estéreo para mezclar al 12% detrás de la locución.
    """
    bgm_urls = [
        "https://cdn.pixabay.com/download/audio/2022/02/07/audio_1808fbf07a.mp3",
        "https://cdn.pixabay.com/download/audio/2022/10/25/audio_946b0939c8.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3"
    ]
    np.random.shuffle(bgm_urls)
    raw_bgm = f"{bgm_path}_raw.mp3"

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
        for url in bgm_urls:
            try:
                res = await client.get(url)
                if res.status_code == 200 and len(res.content) > 25000:
                    with open(raw_bgm, "wb") as f:
                        f.write(res.content)
                    subprocess.run([
                        "ffmpeg", "-y", "-stream_loop", "-1", "-t", str(duration),
                        "-i", raw_bgm, "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", bgm_path
                    ], check=True)
                    return
            except Exception:
                pass

    # Respaldo infalible: Pad ambiental cinematográfico generado nativamente con FFmpeg (acorde La menor profundo)
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"aevalsrc='0.15*sin(2*PI*110*t)+0.12*sin(2*PI*164.81*t)+0.10*sin(2*PI*220*t)+0.08*sin(2*PI*261.63*t)':s=24000:d={duration}",
        "-af", "lowpass=f=450,aecho=0.8:0.88:60:0.4",
        "-c:a", "pcm_s16le", bgm_path
    ], check=True)

# === DESCARGA VISUAL (MODO REALISTA O ANIMADO 3D/ANIME) ===
def is_valid_image(file_path: str) -> bool:
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 5000:
        return False
    with open(file_path, "rb") as f:
        header = f.read(12)
    return header.startswith(b"\xff\xd8") or header.startswith(b"\x89PNG") or (header.startswith(b"RIFF") and b"WEBP" in header)

async def download_valid_image(client: httpx.AsyncClient, term: str, idx: int, img_file: str, w: int, h: int) -> bool:
    clean_term = re.sub(r"[^a-zA-Z0-9 ]", "", term).strip()
    ratio_str = "vertical 9:16 portrait" if h > w else "widescreen 16:9 cinematic"
    unique_seed = np.random.randint(100, 999999)

    if VISUAL_STYLE == "2":
        style_suffix = "3D Pixar Disney animation style, Studio Ghibli vibrant anime lighting, expressive animated characters, cel shaded masterpiece, 8k"
    else:
        style_suffix = "ultra detailed, 8k resolution, photorealistic, dramatic documentary lighting"

    encoded = urllib.parse.quote(f"{clean_term}, {ratio_str}, {style_suffix}")

    urls = [
        f"https://image.pollinations.ai/prompt/{encoded}?width={w}&height={h}&nologo=true&model=flux&seed={unique_seed}",
        f"https://image.pollinations.ai/prompt/{encoded}?width={w}&height={h}&nologo=true&seed={unique_seed + 7}",
        f"https://picsum.photos/seed/{unique_seed}/{w}/{h}"
    ]
    for url in urls:
        try:
            resp = await client.get(url, timeout=30.0)
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

    voice_path = f"output/{prefix}_voice.wav"
    bgm_path = f"output/{prefix}_bgm.wav"
    mixed_audio_path = f"output/{prefix}_mixed.wav"
    ass_path = f"output/{prefix}_subs.ass"
    final_video = f"output/{prefix}_final.mp4"

    # 1. Locución Kokoro-82M + Música de fondo mezclada al 12%
    await generate_voice_kokoro(script_text, voice_path, is_short=is_short)
    total_duration = get_audio_duration(voice_path)

    await prepare_background_music(total_duration, bgm_path)
    fade_start = max(0.0, total_duration - 2.5)
    subprocess.run([
        "ffmpeg", "-y",
        "-i", voice_path,
        "-i", bgm_path,
        "-filter_complex",
        f"[0:a]volume=1.0[v];[1:a]volume=0.12,afade=t=out:st={fade_start}:d=2.5[b];[v][b]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "[aout]",
        "-c:a", "pcm_s16le", mixed_audio_path
    ], check=True)

    # 2. Subtítulos .ASS
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

    # 3. Generación de Escenas (Pexels si es Realista, o 100% Animación IA si VISUAL_STYLE == "2")
    clip_paths = []
    clip_duration = max(total_duration / len(search_terms), 3.0)

    async with httpx.AsyncClient(timeout=45.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
        for i, term in enumerate(search_terms):
            clip_file = f"output/{prefix}_clip_{i}.mp4"
            got_clip = False

            if PEXELS_API_KEY and VISUAL_STYLE == "1":
                try:
                    page_rand = np.random.randint(1, 4)
                    pexels_url = f"https://api.pexels.com/videos/search?query={urllib.parse.quote(term)}&orientation={orientation}&size=medium&per_page=5&page={page_rand}"
                    r = await client.get(pexels_url, headers={"Authorization": PEXELS_API_KEY})
                    videos = r.json().get("videos", [])
                    if videos:
                        chosen_vid = np.random.choice(videos)
                        vfiles = chosen_vid.get("video_files", [])
                        if vfiles:
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
                    scale_w, scale_h = int(w * 1.20), int(h * 1.20)
                    # Alternar entre zoom-in y zoom-out suave para dar sensación de cámara animada viva
                    if i % 2 == 0:
                        z_expr = "min(zoom+0.0016,1.25)"
                    else:
                        z_expr = "if(eq(on,1),1.22,max(1.0,zoom-0.0015))"
                    subprocess.run([
                        "ffmpeg", "-y", "-loop", "1", "-t", str(clip_duration), "-i", img_file,
                        "-vf", f"scale={scale_w}:{scale_h}:force_original_aspect_ratio=increase,crop={scale_w}:{scale_h},zoompan=z='{z_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={w}x{h}:fps=30",
                        "-t", str(clip_duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20", clip_file
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
        "-i", mixed_audio_path,
        "-vf", f"ass={ass_path}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "21",
        "-c:a", "aac", "-b:a", "192k", "-shortest", final_video
    ], check=True)

    return final_video

# === SUBIDA MULTI-CANAL CON PROGRAMACIÓN HORARIA (publishAt) ===
def upload_youtube_rotated(
    video_path: str,
    title: str,
    description: str,
    tags: list[str],
    is_short: bool = True,
    publish_at_iso: str | None = None
) -> str:
    # 1. Buscar tokens específicos del canal seleccionado (ej. tokens/ch2_token_*.json)
    channel_tokens = sorted(glob.glob(f"tokens/{CHANNEL_ID}_token_*.json"))
    # Respaldo automático a cualquier token disponible si aún no se cargó uno exclusivo para ese canal
    if not channel_tokens:
        channel_tokens = sorted(glob.glob("tokens/*.json"))

    if not channel_tokens:
        raise RuntimeError(f"No se encontraron tokens de YouTube para el canal {CHANNEL_ID} en /tokens.")

    status_body = {"selfDeclaredMadeForKids": False}
    if publish_at_iso:
        status_body["privacyStatus"] = "private"
        status_body["publishAt"] = publish_at_iso
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
            response = youtube.videos().insert(part="snippet,status", body=body, media_body=media).execute()
            vid_id = response["id"]
            return f"https://youtube.com/shorts/{vid_id}" if is_short else f"https://youtu.be/{vid_id}"
        except HttpError as e:
            if e.resp.status in [403, 429] and ("quotaExceeded" in str(e) or "uploadLimitExceeded" in str(e)):
                print(f"Cuota agotada en {token_path}, rotando al siguiente token...")
                continue
            raise e
    raise RuntimeError(f"Todos los tokens del canal {CHANNEL_ID.upper()} agotaron su cuota.")

# === ORQUESTADOR PRINCIPAL INTELIGENTE ===
async def main():
    try:
        mode = "both"
        if MODE_CHOICE == "1":
            mode = "short"
        elif MODE_CHOICE == "2":
            mode = "long"
        elif MODE_CHOICE == "3":
            mode = "both"

        style_name = "Animado 3D / Anime (IA)" if VISUAL_STYLE == "2" else "Realista Cinematográfico"

        await send_telegram(
            f"⚙️ Iniciando Fábrica Multi-Canal:\n"
            f"📺 Canal destino: {CHANNEL_ID.upper()}\n"
            f"📌 Nicho base: {RAW_TOPIC}\n"
            f"🎨 Estilo Visual: {style_name}\n"
            f"🎵 Música ambiental: Activa (12%)\n"
            f"📦 Cantidad: {VIDEO_COUNT}\n"
            f"🎬 Modo: {mode.upper()}"
        )

        niche_analysis = await analyze_niche_competitors(RAW_TOPIC)

        await send_telegram(
            f"💡 Estudio de Competencia Completado:\n"
            f"🏆 Estrategia: {niche_analysis['strategy']}\n"
            f"🎯 Tema optimizado: {niche_analysis['topic']}"
        )

        subtopics = await generate_subtopics(niche_analysis['topic'], VIDEO_COUNT)

        for idx, current_topic in enumerate(subtopics, start=1):
            prefix_tag = f"[{idx}/{VIDEO_COUNT}]"
            long_url = "https://youtube.com"
            long_title = current_topic

            # Programación escalonada: el video 1 sale Público de inmediato; los siguientes cada +4 horas
            publish_at_iso = None
            schedule_label = "Público Inmediato"
            if idx > 1:
                sched_dt = datetime.now(timezone.utc) + timedelta(hours=4 * (idx - 1))
                publish_at_iso = sched_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                schedule_label = f"Programado ({publish_at_iso} UTC)"

            effective_topic = f"{current_topic}. Enfoque estratégico: {niche_analysis['hook']}" if idx == 1 and niche_analysis['hook'] else current_topic

            # 1. VIDEO LARGO HORIZONTAL (16:9)
            if mode in ["both", "long"]:
                await send_telegram(f"🎥 {prefix_tag} Creando Video Largo ({style_name}) | {schedule_label}...")
                long_plan = await generate_long_plan(effective_topic)
                long_title = long_plan["title"]

                long_video_path = await render_video(f"long_{idx}", long_plan["script"], long_plan["search_terms"], is_short=False)
                long_url = upload_youtube_rotated(
                    long_video_path, long_title, long_plan["description"], long_plan["tags"],
                    is_short=False, publish_at_iso=publish_at_iso
                )
                await send_telegram(f"✅ {prefix_tag} ¡Video Largo Listo ({schedule_label})!\n📺 {long_title}\n🔗 {long_url}")

            # 2. SHORT VERTICAL (9:16)
            if mode in ["both", "short"]:
                is_standalone = (mode == "short")
                await send_telegram(f"⚡ {prefix_tag} Creando Short ({style_name}) | {schedule_label}...")
                short_plan = await generate_teaser_short_plan(effective_topic, long_title, long_url, is_standalone=is_standalone)

                short_video_path = await render_video(
                    f"short_{idx}",
                    short_plan["script"],
                    short_plan["search_terms"],
                    is_short=True,
                    cta_banner=(mode == "both")
                )
                short_url = upload_youtube_rotated(
                    short_video_path, short_plan["title"], short_plan["description"], short_plan["tags"],
                    is_short=True, publish_at_iso=publish_at_iso
                )
                await send_telegram(f"🔥 {prefix_tag} ¡Short Listo ({schedule_label})!\n🔗 {short_url}")

        await send_telegram(f"🏁 ¡Tanda de {VIDEO_COUNT} producción(es) en {CHANNEL_ID.upper()} finalizada con éxito!")

    except Exception as e:
        await send_telegram(f"❌ Error en la fábrica: {str(e)}")
        raise e

if __name__ == "__main__":
    asyncio.run(main())
