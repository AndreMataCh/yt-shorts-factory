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

try:
    from gradio_client import Client as GradioClient, handle_file
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

def clean_secret(val: str) -> str:
    if not val:
        return ""
    return re.sub(r'[\r\n\t"\'\s]+', '', str(val))

# === DESEMPAQUETADO AUTOMÁTICO DE BUNDLE_JSON ===
RAW_BUNDLE = (os.getenv("BUNDLE_JSON") or "").strip()
BUNDLE = {}
if RAW_BUNDLE:
    try:
        BUNDLE = json.loads(RAW_BUNDLE)
    except Exception as e:
        print(f"Aviso al parsear BUNDLE_JSON: {e}")
        BUNDLE = {}

def get_cfg(env_key: str, bundle_key: str, default: str = "") -> str:
    val = os.getenv(env_key) or BUNDLE.get(bundle_key) or default
    return str(val).strip()

BOT_TOKEN = clean_secret(get_cfg("TELEGRAM_BOT_TOKEN", "telegram_bot_token"))
GROQ_API_KEY = clean_secret(get_cfg("GROQ_API_KEY", "groq_api_key"))
GEMINI_API_KEY = clean_secret(get_cfg("GEMINI_API_KEY", "gemini_api_key"))
OPENROUTER_API_KEY = clean_secret(get_cfg("OPENROUTER_API_KEY", "openrouter_api_key"))
PEXELS_API_KEY = clean_secret(get_cfg("PEXELS_API_KEY", "pexels_api_key"))
HF_TOKEN = clean_secret(get_cfg("HF_TOKEN", "hf_token"))

CHANNEL_ID = get_cfg("CHANNEL_ID", "channel_id", "ch1").lower()
CHANNEL_NAME = get_cfg("CHANNEL_NAME", "channel_name", CHANNEL_ID.upper())
CHANNEL_TOKEN_JSON = get_cfg("CHANNEL_TOKEN_JSON", "channel_token_json", "")

RAW_TOPIC = get_cfg("TOPIC", "topic", "Los secretos mejor guardados de la tecnología")
VISUAL_STYLE = get_cfg("VISUAL_STYLE", "visual_style", "2")
VOICE_STYLE = get_cfg("VOICE_STYLE", "voice_style", "3")
RAW_COUNT = get_cfg("VIDEO_COUNT", "video_count", "1")
MODE_CHOICE = get_cfg("MODE", "mode", "3")

MONETIZATION_URL = get_cfg("MONETIZATION_URL", "monetization_url", "")
CUSTOM_TITLE = get_cfg("CUSTOM_TITLE", "custom_title", "")
CUSTOM_DESCRIPTION = get_cfg("CUSTOM_DESCRIPTION", "custom_description", "")
SOURCE_VIDEO_URL = get_cfg("SOURCE_VIDEO_URL", "source_video_url", "")
CHAT_ID = clean_secret(get_cfg("CHAT_ID", "chat_id", ""))
STATUS_MSG_ID = clean_secret(get_cfg("STATUS_MSG_ID", "status_msg_id", ""))

try:
    VIDEO_COUNT = max(1, min(int(re.sub(r"[^0-9]", "", RAW_COUNT) or "1"), 10))
except ValueError:
    VIDEO_COUNT = 1

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}

os.makedirs("tokens", exist_ok=True)
os.makedirs("assets/voices", exist_ok=True)
os.makedirs("assets/avatars", exist_ok=True)

if CHANNEL_TOKEN_JSON:
    try:
        parsed_tok = json.loads(CHANNEL_TOKEN_JSON)
        with open(f"tokens/{CHANNEL_ID}_token_kv.json", "w", encoding="utf-8") as f:
            json.dump(parsed_tok, f)
        print(f"Token OAuth cargado para el canal: {CHANNEL_NAME} ({CHANNEL_ID})")
    except Exception as e:
        print(f"Aviso al parsear CHANNEL_TOKEN_JSON: {e}")

print("Inicializando motor neuronal base Kokoro-82M...")
kokoro_pipeline = KPipeline(lang_code='e')

def run_ffmpeg(cmd: list[str], step_desc: str = "FFmpeg"):
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        err_msg = res.stderr.decode("utf-8", errors="ignore")
        tail_err = "\n".join(err_msg.splitlines()[-15:])
        print(f"❌ Error en {step_desc} (Código {res.returncode}):\n{tail_err}")
        raise RuntimeError(f"Fallo en {step_desc} (Exit {res.returncode}): {tail_err[-250:]}")
    return res

async def send_telegram(text: str):
    print(text)
    if not BOT_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            await client.post(url, json={
                "chat_id": int(CHAT_ID),
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": False
            })
        except Exception as e:
            print(f"Aviso Telegram sendMessage: {e}")

async def update_live_progress(percent: int, stage: str, detail: str = ""):
    global STATUS_MSG_ID
    print(f"[{percent}%] {stage} - {detail}")
    if not BOT_TOKEN or not CHAT_ID:
        return

    filled = max(0, min(10, round(percent / 10)))
    bar = "🟩" * filled + "⬜" * (10 - filled)
    text = (
        f"🎬 <b>Fábrica en Ejecución — {CHANNEL_NAME}</b>\n"
        f"{bar} <b>{percent}%</b>\n\n"
        f"📌 <b>Tema:</b> {RAW_TOPIC}\n"
        f"⚡ <b>Estado actual:</b> {stage}\n"
        f"<i>{detail}</i>"
    )

    async with httpx.AsyncClient(timeout=15.0) as client:
        if STATUS_MSG_ID:
            try:
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/editMessageText"
                r = await client.post(url, json={
                    "chat_id": int(CHAT_ID),
                    "message_id": int(STATUS_MSG_ID),
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True
                })
                if r.status_code == 200:
                    return
            except Exception as e:
                print(f"Aviso editMessageText: {e}")

        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            r = await client.post(url, json={
                "chat_id": int(CHAT_ID),
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True
            })
            if r.status_code == 200:
                data = r.json()
                if data.get("ok") and data.get("result", {}).get("message_id"):
                    STATUS_MSG_ID = str(data["result"]["message_id"])
        except Exception as e:
            print(f"Aviso creando panel de progreso: {e}")

def clean_llm_output(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()

async def ask_llm(prompt: str, max_tokens: int = 1200) -> str:
    errors = []

    if GEMINI_API_KEY and GEMINI_API_KEY.startswith("AIza"):
        gemini_headers = {**BROWSER_HEADERS, "x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=60.0, headers=gemini_headers) as client:
            for model in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest"]:
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

    if GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_"):
        groq_headers = {**BROWSER_HEADERS, "Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=60.0, headers=groq_headers) as client:
            for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
                try:
                    res = await client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": max_tokens}
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

    if OPENROUTER_API_KEY:
        try:
            async with httpx.AsyncClient(
                timeout=60.0,
                headers={**BROWSER_HEADERS, "Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
            ) as client:
                res = await client.post("https://openrouter.ai/api/v1/chat/completions", json={
                    "model": "meta-llama/llama-3.3-70b-instruct:free",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                })
                if res.status_code == 200:
                    content = res.json()["choices"][0]["message"].get("content", "")
                    cleaned = clean_llm_output(content)
                    if cleaned:
                        return cleaned
        except Exception as e:
            errors.append(f"OpenRouter: {str(e)[:60]}")

    async with httpx.AsyncClient(timeout=60.0, headers={**BROWSER_HEADERS, "Content-Type": "application/json"}, follow_redirects=True) as client:
        for poly_model in ["openai", "mistral"]:
            try:
                res = await client.post(
                    "https://text.pollinations.ai/openai",
                    json={"model": poly_model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
                )
                if res.status_code == 200:
                    data = res.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0].get("message", {}).get("content", "")
                        cleaned = clean_llm_output(content)
                        if cleaned:
                            return cleaned
            except Exception as e:
                errors.append(f"Pollinations ({poly_model}): {str(e)[:60]}")

    raise RuntimeError(f"Falla total de LLMs: {' | '.join(errors)[:350]}")

def get_seo_context() -> str:
    if os.path.exists("seo_context.md"):
        try:
            with open("seo_context.md", "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return f"\n\nContexto actual de tendencias web analizadas:\n{content[:1500]}\n"
        except Exception:
            pass
    return ""

def parse_script_and_sfx(raw_script: str) -> tuple[str, list[dict]]:
    pattern = re.compile(r"\[SFX:\s*(whoosh|impact|riser|glitch)\]", flags=re.IGNORECASE)
    tokens = raw_script.split()
    clean_words = []
    sfx_cues = []

    for token in tokens:
        match = pattern.search(token)
        if match:
            sfx_type = match.group(1).lower()
            sfx_cues.append({"type": sfx_type, "word_index": len(clean_words)})
            rem = pattern.sub("", token).strip()
            if rem:
                clean_words.append(rem)
        else:
            clean_t = re.sub(r"[\*\#\[\]\(\)]", "", token).strip()
            if clean_t:
                clean_words.append(clean_t)

    if not sfx_cues and len(clean_words) > 10:
        sfx_cues.append({"type": "impact", "word_index": 0})
        sfx_cues.append({"type": "whoosh", "word_index": len(clean_words) // 3})
        sfx_cues.append({"type": "riser", "word_index": (len(clean_words) * 2) // 3})

    return " ".join(clean_words), sfx_cues

async def generate_long_plan(topic: str) -> dict:
    seo_intel = get_seo_context()
    script_prompt = (
        f"Escribe un guion documental fascinante y dinámico en español neutro de unas 350 a 450 palabras sobre: '{topic}'. "
        f"{seo_intel}"
        "Inserta entre 5 y 8 marcadores de efectos de sonido con el formato exacto [SFX: impact], [SFX: whoosh], [SFX: riser] o [SFX: glitch] en momentos clave. "
        "REGLAS: Devuelve ÚNICAMENTE el texto hablado con sus etiquetas [SFX: ...]. Sin asteriscos, sin títulos, sin emojis."
    )
    raw_script = await ask_llm(script_prompt, max_tokens=1400)
    clean_script, sfx_cues = parse_script_and_sfx(raw_script)

    title = CUSTOM_TITLE or (await ask_llm(f"Crea un título magnético de YouTube (máx 65 caracteres) sobre '{topic}'. SOLO el título sin comillas.", max_tokens=150)).replace('"', '').splitlines()[0].strip()[:75]

    terms_raw = await ask_llm(
        f"Based on this narrative about '{topic}': '{clean_script[:500]}', provide 14 chronological visual scene prompts in English (4 to 7 words each, cinematic lighting, specific action). Return ONLY the 14 comma-separated prompts.",
        max_tokens=350
    )
    terms = [t.strip() for t in terms_raw.split(",") if t.strip()][:14]
    while len(terms) < 14:
        terms.append(f"cinematic dramatic scene {topic}")

    monetization_block = f"\n\n🔥 Descarga la Plantilla / Recurso Oficial aquí:\n👉 {MONETIZATION_URL}" if MONETIZATION_URL else ""
    base_desc = CUSTOM_DESCRIPTION or f"{title}\n\nExploramos a fondo: {topic}.\n\n{clean_script[:250]}..."

    return {
        "title": title,
        "description": f"{base_desc}{monetization_block}\n\n#Documental #Tecnologia #Viral",
        "tags": ["documental", "tecnologia", "ciencia", "viral"],
        "script": clean_script,
        "sfx_cues": sfx_cues,
        "search_terms": terms
    }

async def generate_teaser_short_plan(topic: str, long_title: str = "", long_url: str = "") -> dict:
    seo_intel = get_seo_context()
    if MONETIZATION_URL:
        cta_line = "Descarga la guía oficial y mira el video completo en la descripción."
    elif long_url:
        cta_line = "Descubre el caso completo en el enlace de la descripción."
    else:
        cta_line = "Suscríbete ahora para descubrir más secretos ocultos."

    script_prompt = (
        f"Escribe un guion de YouTube Short ultradinámico de 30 segundos sobre '{topic}'. "
        f"{seo_intel}"
        f"Inserta 4 marcadores [SFX: impact], [SFX: whoosh], [SFX: riser] o [SFX: glitch] y termina diciendo: '{cta_line}' "
        "REGLAS: Devuelve ÚNICAMENTE el texto hablado en español neutro con las etiquetas [SFX: ...], sin asteriscos ni emojis."
    )
    raw_script = await ask_llm(script_prompt, max_tokens=600)
    clean_script, sfx_cues = parse_script_and_sfx(raw_script)

    title = (await ask_llm(f"Crea un título corto y viral (máx 55 caracteres) para un Short sobre '{topic}'. SOLO el título sin comillas.", max_tokens=150)).replace('"', '').splitlines()[0].strip()[:65]

    terms_raw = await ask_llm(
        f"Based on this short story about '{topic}': '{clean_script}', provide 10 chronological visual scene descriptions in English (4 to 7 words each). Return ONLY the 10 comma-separated descriptions.",
        max_tokens=280
    )
    terms = [t.strip() for t in terms_raw.split(",") if t.strip()][:10]
    while len(terms) < 10:
        terms.append("cinematic suspenseful dark scene")

    desc_link = f"\n\n🔗 Video completo: {long_url}" if long_url else ""
    monetization_block = f"\n📦 Plantilla / Recurso Oficial: {MONETIZATION_URL}" if MONETIZATION_URL else ""
    return {
        "title": title,
        "description": f"{title}{desc_link}{monetization_block}\n\n#Shorts #Viral",
        "tags": ["shorts", "viral", "documental"],
        "script": clean_script,
        "sfx_cues": sfx_cues,
        "search_terms": terms
    }

def format_ass_time(seconds: float) -> str:
    total_cs = int(round(max(0.0, float(seconds)) * 100))
    h = total_cs // 360000
    m = (total_cs % 360000) // 6000
    s = (total_cs % 6000) // 100
    cs = total_cs % 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def get_audio_duration(audio_path: str) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ]).decode().strip()
    return float(out)

async def generate_voice_hybrid(script_text: str, audio_path: str, is_short: bool = True):
    raw_voice = f"{audio_path}_raw.wav"
    cloned_ok = False

    ref_candidates = [
        f"assets/voices/{CHANNEL_ID}.wav",
        "assets/voices/default.wav"
    ]
    ref_audio = next((p for p in ref_candidates if os.path.exists(p)), None)

    if HAS_GRADIO and ref_audio:
        for space_id in ["jpgallegoar/Spanish-F5", "mrfakename/E2-F5-TTS"]:
            try:
                print(f"🎙️ Clonando voz con F5-TTS ({space_id})...")
                client = GradioClient(space_id, hf_token=HF_TOKEN if HF_TOKEN else None)
                res = await asyncio.to_thread(
                    client.predict,
                    handle_file(ref_audio),
                    "",
                    script_text,
                    True,
                    api_name="/infer"
                )
                out_file = res[0] if isinstance(res, (list, tuple)) else res
                if out_file and os.path.exists(str(out_file)):
                    run_ffmpeg([
                        "ffmpeg", "-y", "-i", str(out_file),
                        "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", raw_voice
                    ], "Conversión F5-TTS")
                    cloned_ok = True
                    break
            except Exception as e:
                print(f"Aviso F5-TTS ({space_id}): {e}")

    if not cloned_ok:
        voice_map = {"1": "em_alex", "2": "ef_dora", "3": "em_santa"}
        voice_name = voice_map.get(VOICE_STYLE, "em_santa")
        speed_val = 1.12 if is_short else 1.04
        generator = kokoro_pipeline(script_text, voice=voice_name, speed=speed_val)
        audio_chunks = [chunk for _, _, chunk in generator]
        if not audio_chunks:
            raise RuntimeError("Kokoro no generó audio.")
        complete_audio = np.concatenate(audio_chunks)
        sf.write(raw_voice, complete_audio, 24000)

    filtro_voz = (
        "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-48dB:"
        "stop_periods=-1:stop_duration=0.22:stop_threshold=-48dB,"
        "equalizer=f=180:t=q:w=1:g=2,equalizer=f=3200:t=q:w=1:g=3,"
        "acompressor=threshold=-16dB:ratio=3:attack=5:release=50"
    )
    run_ffmpeg([
        "ffmpeg", "-y", "-i", raw_voice,
        "-af", filtro_voz,
        "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", audio_path
    ], "Limpieza y masterización de voz")

def ensure_sfx_bank() -> dict[str, str]:
    os.makedirs("output/sfx", exist_ok=True)
    bank = {
        "whoosh": "output/sfx/whoosh.wav",
        "impact": "output/sfx/impact.wav",
        "riser": "output/sfx/riser.wav",
        "glitch": "output/sfx/glitch.wav"
    }
    if not os.path.exists(bank["whoosh"]):
        run_ffmpeg([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "anoisesrc=d=0.45:c=pink:r=24000:a=0.5",
            "-af", "bandpass=f=900:width_type=h:w=700,afade=t=in:ss=0:d=0.18,afade=t=out:st=0.22:d=0.23",
            "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", bank["whoosh"]
        ], "SFX Whoosh")
    if not os.path.exists(bank["impact"]):
        run_ffmpeg([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "aevalsrc='0.8*sin(2*PI*(110*exp(-7*t))*t)*exp(-4.5*t)':s=24000:d=0.85",
            "-af", "lowpass=f=220", "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", bank["impact"]
        ], "SFX Impact")
    if not os.path.exists(bank["riser"]):
        run_ffmpeg([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "aevalsrc='0.35*sin(2*PI*(140+380*t*t)*t)':s=24000:d=1.2",
            "-af", "afade=t=in:ss=0:d=1.0,afade=t=out:st=1.05:d=0.15",
            "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", bank["riser"]
        ], "SFX Riser")
    if not os.path.exists(bank["glitch"]):
        run_ffmpeg([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "aevalsrc='0.4*sgn(sin(2*PI*420*t))*sin(2*PI*65*t)':s=24000:d=0.22",
            "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", bank["glitch"]
        ], "SFX Glitch")
    return bank

def resolve_sfx_timestamps(sfx_cues: list[dict], words_timed: list[dict], total_duration: float) -> list[dict]:
    resolved = []
    if not words_timed:
        return resolved
    for cue in sfx_cues:
        idx = min(max(0, cue["word_index"]), len(words_timed) - 1)
        t_sec = max(0.0, float(words_timed[idx]["start"]) - 0.05)
        if t_sec < total_duration - 0.2:
            resolved.append({"type": cue["type"], "time": t_sec})
    return resolved[:10]

async def prepare_channel_avatar(prefix: str, voice_path: str) -> str:
    avatar_candidates = [
        f"assets/avatars/{CHANNEL_ID}.png",
        "assets/avatars/default.png"
    ]
    avatar_img = next((p for p in avatar_candidates if os.path.exists(p)), f"assets/avatars/{CHANNEL_ID}.png")

    if not os.path.exists(avatar_img):
        prompt = urllib.parse.quote(f"cinematic portrait of charismatic host for {CHANNEL_NAME}, studio lighting, looking at camera, 8k")
        url = f"https://image.pollinations.ai/prompt/{prompt}?width=512&height=512&nologo=true"
        try:
            async with httpx.AsyncClient(timeout=25.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
                r = await client.get(url)
                if r.status_code == 200 and len(r.content) > 5000:
                    with open(avatar_img, "wb") as f:
                        f.write(r.content)
        except Exception:
            pass

    if not os.path.exists(avatar_img):
        return ""

    avatar_vid = f"output/{prefix}_avatar.mp4"
    hook_audio = f"output/{prefix}_hook.wav"
    run_ffmpeg([
        "ffmpeg", "-y", "-t", "4.5", "-i", voice_path,
        "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", hook_audio
    ], "Hook Audio Avatar")

    if HAS_GRADIO:
        for space_id in ["KwaiVGI/LivePortrait", "TMElyralab/MuseTalk"]:
            try:
                client = GradioClient(space_id, hf_token=HF_TOKEN if HF_TOKEN else None)
                res = await asyncio.to_thread(
                    client.predict,
                    handle_file(avatar_img),
                    handle_file(hook_audio),
                    api_name="/predict"
                )
                out_v = res[0] if isinstance(res, (list, tuple)) else res
                if out_v and os.path.exists(str(out_v)):
                    run_ffmpeg([
                        "ffmpeg", "-y", "-t", "4.5", "-i", str(out_v),
                        "-vf", "scale=320:320:force_original_aspect_ratio=increase,crop=320:320,fps=30",
                        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", avatar_vid
                    ], "LivePortrait Avatar")
                    return avatar_vid
            except Exception:
                pass

    run_ffmpeg([
        "ffmpeg", "-y", "-loop", "1", "-t", "4.5", "-i", avatar_img,
        "-vf", "scale=360:360,zoompan=z='1.04+0.03*sin(2*PI*on/45)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=135:s=300x300:fps=30,drawbox=x=0:y=0:w=300:h=300:color=0x00FFFF@0.85:t=6",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", avatar_vid
    ], "Fallback Avatar PiP")
    return avatar_vid

async def get_word_timestamps(audio_path: str, script_text: str, total_duration: float) -> list[dict]:
    if GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_"):
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                with open(audio_path, "rb") as f:
                    files = {"file": ("voice.wav", f, "audio/wav")}
                    data = {
                        "model": "whisper-large-v3-turbo",
                        "response_format": "verbose_json",
                        "timestamp_granularities[]": "word",
                        "language": "es"
                    }
                    res = await client.post(
                        "https://api.groq.com/openai/v1/audio/transcriptions",
                        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                        files=files,
                        data=data
                    )
                if res.status_code == 200:
                    words_data = res.json().get("words", [])
                    aligned = []
                    for w in words_data:
                        txt = (w.get("word") or "").strip()
                        if txt:
                            aligned.append({
                                "word": txt,
                                "start": float(w.get("start", 0.0)),
                                "end": float(w.get("end", 0.0))
                            })
                    if len(aligned) >= 3:
                        return aligned
        except Exception:
            pass

    raw_words = [w.strip() for w in script_text.split() if w.strip()]
    if not raw_words:
        return [{"word": "VIDEO", "start": 0.0, "end": total_duration}]

    weights = []
    for w in raw_words:
        base_w = max(len(re.sub(r"[^a-zA-ZáéíóúÁÉÍÓÚñÑ0-9]", "", w)), 1)
        if w.endswith((".", "!", "?", ":")):
            base_w += 3.5
        elif w.endswith((",", ";")):
            base_w += 1.8
        weights.append(float(base_w))

    total_weight = sum(weights)
    current_t = 0.0
    fallback_aligned = []
    for w, weight in zip(raw_words, weights):
        dur = (weight / total_weight) * total_duration
        fallback_aligned.append({
            "word": w,
            "start": current_t,
            "end": min(current_t + dur, total_duration)
        })
        current_t += dur
    return fallback_aligned

def get_channel_subtitle_profile(channel_name: str, topic: str) -> dict:
    text_sig = (channel_name + " " + topic).lower()
    if any(k in text_sig for k in ['finan', 'cripto', 'dinero', 'banco', 'invers', 'negocio', 'bitcoin']):
        return {
            "primary_style": "&H0000D7FF", "active_style": "&H0000FF7F", "outline_style": "&H00000000",
            "primary_tag": "&H00D7FF&", "active_tag": "&H00FF7F&", "font_name": "DejaVu Sans",
            "font_size_short": 82, "font_size_long": 58
        }
    elif any(k in text_sig for k in ['terror', 'misteri', 'oscuro', 'crimen', 'miedo', 'secreto', 'paranormal', 'interrogatorio']):
        return {
            "primary_style": "&H00FFFFFF", "active_style": "&H000000FF", "outline_style": "&H00000000",
            "primary_tag": "&HFFFFFF&", "active_tag": "&H0000FF&", "font_name": "DejaVu Sans",
            "font_size_short": 84, "font_size_long": 60
        }
    return {
        "primary_style": "&H00FFFFFF", "active_style": "&H0000FFFF", "outline_style": "&H00000000",
        "primary_tag": "&HFFFFFF&", "active_tag": "&H00FFFF&", "font_name": "DejaVu Sans",
        "font_size_short": 82, "font_size_long": 58
    }

def sanitize_ass_word(word: str) -> str:
    cleaned = re.sub(r"[\{\}\\\r\n]", "", word).strip().upper()
    return cleaned if cleaned else "..."

def build_viral_ass_subtitles(words_timed: list[dict], ass_path: str, w: int, h: int, is_short: bool, total_duration: float, channel_name: str = "", topic: str = ""):
    profile = get_channel_subtitle_profile(channel_name, topic)
    font_size = profile["font_size_short"] if is_short else profile["font_size_long"]
    margin_v = 420 if is_short else 90
    group_size = 3 if is_short else 5

    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: ViralSub,{profile["font_name"]},{font_size},{profile["primary_style"]},{profile["active_style"]},{profile["outline_style"]},&H80000000,-1,0,0,0,100,100,1,0,1,6,3,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [ass_header]

    for i in range(0, len(words_timed), group_size):
        chunk = words_timed[i:i + group_size]
        if not chunk:
            continue

        for active_idx, active_word in enumerate(chunk):
            start_sec = float(active_word["start"])
            if active_idx + 1 < len(chunk):
                end_sec = max(float(active_word["end"]), float(chunk[active_idx + 1]["start"]))
            elif i + group_size < len(words_timed):
                end_sec = max(float(active_word["end"]), min(float(words_timed[i + group_size]["start"]), float(active_word["end"]) + 0.35))
            else:
                end_sec = min(total_duration, max(float(active_word["end"]) + 0.25, float(active_word["end"])))

            if end_sec <= start_sec:
                end_sec = start_sec + 0.15

            styled_tokens = []
            for idx_in_chunk, item in enumerate(chunk):
                clean_w = sanitize_ass_word(item["word"])
                if idx_in_chunk == active_idx:
                    styled_tokens.append(f"{{\\c{profile['active_tag']}}}{clean_w}{{\\c{profile['primary_tag']}}}")
                else:
                    styled_tokens.append(clean_w)

            pop_tag = r"{\fscx118\fscy118\t(0,110,\fscx100\fscy100)}" if active_idx == 0 else r"{\fscx106\fscy106\t(0,80,\fscx100\fscy100)}"
            text_line = pop_tag + " ".join(styled_tokens)
            lines.append(f"Dialogue: 0,{format_ass_time(start_sec)},{format_ass_time(end_sec)},ViralSub,,0,0,0,,{text_line}\n")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

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
                    run_ffmpeg([
                        "ffmpeg", "-y", "-stream_loop", "-1", "-t", f"{duration:.2f}",
                        "-i", raw_bgm, "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", bgm_path
                    ], "BGM")
                    return
            except Exception:
                pass

    run_ffmpeg([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"aevalsrc='0.12*sin(2*PI*130.81*t)+0.08*sin(2*PI*196*t)':s=24000:d={duration:.2f}",
        "-af", "lowpass=f=600", "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", bgm_path
    ], "BGM Sintético")

def mix_master_audio_with_sfx(voice_path: str, bgm_path: str, resolved_sfx: list[dict], sfx_bank: dict[str, str], total_duration: float, mixed_audio: str):
    fade_start = max(0.0, total_duration - 2.0)
    inputs = ["-i", voice_path, "-i", bgm_path]
    filter_parts = [
        "[0:a]volume=1.18[v]",
        f"[1:a]volume=0.09,afade=t=out:st={fade_start:.2f}:d=2[b]"
    ]
    mix_labels = ["[v]", "[b]"]

    for idx, sfx_item in enumerate(resolved_sfx):
        sfx_file = sfx_bank.get(sfx_item["type"], sfx_bank["whoosh"])
        delay_ms = int(round(sfx_item["time"] * 1000))
        input_idx = idx + 2
        inputs.extend(["-i", sfx_file])
        label = f"[s{idx}]"
        filter_parts.append(f"[{input_idx}:a]volume=0.38,adelay={delay_ms}|{delay_ms}{label}")
        mix_labels.append(label)

    total_inputs = len(mix_labels)
    filter_parts.append(
        f"{''.join(mix_labels)}amix=inputs={total_inputs}:duration=first:dropout_transition=0:normalize=0[aout]"
    )

    run_ffmpeg([
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[aout]", "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", mixed_audio
    ], "Mezcla Multicapa")

def postproducir_video_maestro(input_video: str, output_video: str, is_short: bool = True, trim_silences: bool = True):
    w, h = (1080, 1920) if is_short else (1920, 1080)
    filtro_audio = (
        "silenceremove=stop_periods=-1:stop_duration=0.35:stop_threshold=-48dB,loudnorm=I=-16:TP=-1.5:LRA=11,acompressor"
        if trim_silences else "loudnorm=I=-16:TP=-1.5:LRA=11"
    )
    filtro_video = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps=30"

    run_ffmpeg([
        "ffmpeg", "-y", "-i", input_video,
        "-af", filtro_audio,
        "-vf", filtro_video,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        output_video
    ], "Masterización Final")
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
    return postproducir_video_maestro(raw_input, final_video, is_short=is_short, trim_silences=True)

async def render_video_with_real_clips(prefix: str, script_text: str, search_terms: list[str], is_short: bool = True, sfx_cues: list[dict] = None) -> str:
    os.makedirs("output", exist_ok=True)
    w, h = (1080, 1920) if is_short else (1920, 1080)
    orientation = "portrait" if is_short else "landscape"
    format_label = "Short 9:16" if is_short else "Documental 16:9"

    voice_path = f"output/{prefix}_voice.wav"
    bgm_path = f"output/{prefix}_bgm.wav"
    mixed_audio = f"output/{prefix}_master.wav"
    ass_path = f"output/{prefix}_subs.ass"
    raw_output = f"output/{prefix}_raw.mp4"
    final_video = f"output/{prefix}_final.mp4"

    await update_live_progress(68, f"Clonando voz y alineando fonemas ({format_label})...", "F5-TTS / Kokoro-82M + Whisper Alignment")
    await generate_voice_hybrid(script_text, voice_path, is_short=is_short)
    total_duration = max(1.0, get_audio_duration(voice_path))

    words_timed = await get_word_timestamps(voice_path, script_text, total_duration)
    build_viral_ass_subtitles(words_timed, ass_path, w, h, is_short, total_duration, CHANNEL_NAME, RAW_TOPIC)

    await update_live_progress(76, f"Sintetizando efectos SFX y música ({format_label})...", "Mezcla multicapa sincronizada al milisegundo")
    sfx_bank = ensure_sfx_bank()
    resolved_sfx = resolve_sfx_timestamps(sfx_cues or [], words_timed, total_duration)
    await prepare_background_music(total_duration, bgm_path)
    mix_master_audio_with_sfx(voice_path, bgm_path, resolved_sfx, sfx_bank, total_duration, mixed_audio)

    target_cut_sec = 2.8 if is_short else 3.8
    num_cuts = max(int(np.ceil(total_duration / target_cut_sec)), len(search_terms), 4)
    clip_duration = total_duration / num_cuts
    frames_per_clip = max(30, int(round(clip_duration * 30)))

    await update_live_progress(84, f"Generando {num_cuts} escenas visuales y cámara fluida ({format_label})...", "Flux AI / Pexels HD + VFX Flash")

    clip_paths = []
    used_video_ids = set()

    async with httpx.AsyncClient(timeout=35.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
        for i in range(num_cuts):
            term = search_terms[i % len(search_terms)]
            clip_file = f"output/{prefix}_clip_{i}.mp4"
            got_clip = False
            vfx_flash = "fade=t=in:st=0:d=0.12,eq=contrast=1.08:saturation=1.15,unsharp=5:5:0.8:3:3:0.4"

            if VISUAL_STYLE == "1" and PEXELS_API_KEY:
                try:
                    pexels_url = f"https://api.pexels.com/videos/search?query={urllib.parse.quote(term)}&orientation={orientation}&per_page=8"
                    r = await client.get(pexels_url, headers={"Authorization": PEXELS_API_KEY})
                    videos = [v for v in r.json().get("videos", []) if v.get("id") not in used_video_ids] or r.json().get("videos", [])
                    if videos:
                        chosen = videos[i % len(videos)]
                        used_video_ids.add(chosen.get("id"))
                        vfiles = chosen.get("video_files", [])
                        if vfiles:
                            hd_files = [vf for vf in vfiles if 720 <= vf.get("width", 0) <= 1920] or vfiles
                            best = max(hd_files, key=lambda x: x.get("width", 0))
                            vid_data = await client.get(best["link"])
                            raw_vid = f"output/{prefix}_source_{i}.mp4"
                            with open(raw_vid, "wb") as f:
                                f.write(vid_data.content)
                            run_ffmpeg([
                                "ffmpeg", "-y", "-stream_loop", "-1", "-t", f"{clip_duration:.2f}", "-i", raw_vid,
                                "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,{vfx_flash},fps=30",
                                "-r", "30", "-vsync", "cfr", "-video_track_timescale", "15360",
                                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", clip_file
                            ], f"Clip Pexels {i}")
                            got_clip = True
                except Exception:
                    pass

            if not got_clip:
                try:
                    ai_prompt = urllib.parse.quote(f"cinematic hyperrealistic scene, {term}, dramatic lighting, 8k, film grain")
                    img_url = f"https://image.pollinations.ai/prompt/{ai_prompt}?width={w}&height={h}&seed={i + 100}&nologo=true"
                    img_res = await client.get(img_url)
                    if img_res.status_code == 200 and len(img_res.content) > 5000:
                        scene_img = f"output/{prefix}_scene_{i}.jpg"
                        with open(scene_img, "wb") as f:
                            f.write(img_res.content)
                        z_expr = "min(zoom+0.0018,1.18)" if i % 2 == 0 else "if(eq(on,1),1.18,max(1.0,zoom-0.0018))"
                        run_ffmpeg([
                            "ffmpeg", "-y", "-loop", "1", "-t", f"{clip_duration:.2f}", "-i", scene_img,
                            "-vf", f"scale={int(w*1.2)}:{int(h*1.2)},zoompan=z='{z_expr}':x='iw/2-(iw/zoom/2)+12*sin(on/20)':y='ih/2-(ih/zoom/2)':d={frames_per_clip}:s={w}x{h}:fps=30,setsar=1,{vfx_flash}",
                            "-r", "30", "-vsync", "cfr", "-video_track_timescale", "15360",
                            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", clip_file
                        ], f"Escena IA {i}")
                        got_clip = True
                except Exception:
                    pass

            if not got_clip:
                run_ffmpeg([
                    "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=0x0f172a:s={w}x{h}:d={clip_duration:.2f}:r=30",
                    "-vf", "setsar=1", "-r", "30", "-vsync", "cfr", "-video_track_timescale", "15360",
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", clip_file
                ], f"Clip Fallback {i}")

            clip_paths.append(os.path.abspath(clip_file).replace("\\", "/"))

    concat_file = f"output/{prefix}_concat.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for cp in clip_paths:
            f.write(f"file '{cp}'\n")

    await update_live_progress(92, f"Incrustando Avatar PiP y Subtítulos ASS ({format_label})...", "Masterización final en FFmpeg a 30fps")
    avatar_vid = await prepare_channel_avatar(prefix, voice_path)
    ass_filter_path = ass_path.replace("\\", "/").replace(":", "\\:")

    if avatar_vid and os.path.exists(avatar_vid):
        pip_y = h - 380 if is_short else h - 340
        run_ffmpeg([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_file,
            "-i", mixed_audio,
            "-i", avatar_vid,
            "-filter_complex",
            f"[0:v][2:v]overlay=x=40:y={pip_y}:enable='between(t,0,4.5)'[vpip];[vpip]ass={ass_filter_path}[vout]",
            "-map", "[vout]", "-map", "1:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-shortest", raw_output
        ], "Render con Avatar PiP")
    else:
        run_ffmpeg([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-i", mixed_audio, "-vf", f"ass={ass_filter_path}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-shortest", raw_output
        ], "Render Estándar")

    postproducir_video_maestro(raw_output, final_video, is_short=is_short, trim_silences=False)
    return final_video

def upload_youtube_rotated(video_path: str, title: str, description: str, tags: list[str], is_short: bool = True) -> str:
    channel_tokens = sorted(glob.glob(f"tokens/{CHANNEL_ID}_token_*.json")) or sorted(glob.glob("tokens/*.json"))
    if not channel_tokens:
        raise RuntimeError(f"No se encontraron tokens OAuth para el canal '{CHANNEL_NAME}' ({CHANNEL_ID}).")

    status_body = {"selfDeclaredMadeForKids": False, "privacyStatus": "public"}

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
        await update_live_progress(58, "Redactando guion documental y prompts visuales...", f"Canal: {CHANNEL_NAME} | Videos: {VIDEO_COUNT}")

        if SOURCE_VIDEO_URL:
            is_short_mode = (MODE_CHOICE == "1")
            edited_vid = await edit_uploaded_raw_video(SOURCE_VIDEO_URL, "raw_edit", is_short=is_short_mode)
            title = CUSTOM_TITLE or RAW_TOPIC[:80]
            desc = CUSTOM_DESCRIPTION or f"{title}\n\n{MONETIZATION_URL}"
            vid_url = upload_youtube_rotated(edited_vid, title, desc, ["viral", "pro"], is_short=is_short_mode)
            await update_live_progress(100, "¡Video Pregrabado Publicado! ✅", vid_url)
            await send_telegram(f"✅ <b>Video Pregrabado Masterizado y Publicado:</b>\n🔗 {vid_url}")
            return

        if VIDEO_COUNT > 1:
            variation_prompt = f"Para el nicho '{RAW_TOPIC}', genera {VIDEO_COUNT} subtemas virales. Devuelve ÚNICAMENTE los títulos separados por punto y coma (;)."
            raw_subtopics = await ask_llm(variation_prompt, max_tokens=300)
            subtopics = [s.strip() for s in raw_subtopics.split(";") if s.strip()][:VIDEO_COUNT]
            while len(subtopics) < VIDEO_COUNT:
                subtopics.append(f"{RAW_TOPIC} - Parte {len(subtopics)+1}")
        else:
            subtopics = [RAW_TOPIC]

        published_summary = []

        for idx, topic in enumerate(subtopics, start=1):
            long_url = ""
            long_title = topic

            if MODE_CHOICE in ["2", "3"]:
                await update_live_progress(62, f"Diseñando Video Largo ({idx}/{len(subtopics)})...", topic)
                long_plan = await generate_long_plan(topic)
                long_title = long_plan["title"]
                long_vid = await render_video_with_real_clips(
                    f"long_{idx}", long_plan["script"], long_plan["search_terms"],
                    is_short=False, sfx_cues=long_plan.get("sfx_cues", [])
                )
                await update_live_progress(96, f"Subiendo Video Largo ({idx}/{len(subtopics)}) a YouTube...", long_plan["title"])
                long_url = upload_youtube_rotated(long_vid, long_plan["title"], long_plan["description"], long_plan["tags"], is_short=False)
                published_summary.append(f"🖥️ <b>Largo:</b> <a href='{long_url}'>{long_plan['title']}</a>")

            if MODE_CHOICE in ["1", "3"]:
                await update_live_progress(65, f"Diseñando YouTube Short ({idx}/{len(subtopics)})...", topic)
                short_plan = await generate_teaser_short_plan(topic, long_title, long_url)
                short_vid = await render_video_with_real_clips(
                    f"short_{idx}", short_plan["script"], short_plan["search_terms"],
                    is_short=True, sfx_cues=short_plan.get("sfx_cues", [])
                )
                await update_live_progress(98, f"Subiendo Short ({idx}/{len(subtopics)}) a YouTube...", short_plan["title"])
                short_url = upload_youtube_rotated(short_vid, short_plan["title"], short_plan["description"], short_plan["tags"], is_short=True)
                published_summary.append(f"📱 <b>Short:</b> <a href='{short_url}'>{short_plan['title']}</a>")

        links_block = "\n".join(published_summary)
        gum_line = f"\n💰 <b>Producto Digital / Gumroad:</b> {MONETIZATION_URL}" if MONETIZATION_URL else ""
        await update_live_progress(100, "¡Producción y Publicación Completada! 🏁", f"Se publicaron {len(published_summary)} activos en {CHANNEL_NAME}.")
        await send_telegram(
            f"🏁 <b>¡Tanda Finalizada con Éxito en {CHANNEL_NAME}!</b>\n\n"
            f"{links_block}{gum_line}"
        )
    except Exception as e:
        await update_live_progress(100, "❌ Error durante la ejecución", str(e)[:180])
        await send_telegram(f"❌ <b>Error crítico en fábrica ({CHANNEL_NAME}):</b>\n<code>{str(e)}</code>")
        raise e

if __name__ == "__main__":
    asyncio.run(main())
