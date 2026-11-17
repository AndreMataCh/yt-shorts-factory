import os
import re
import asyncio
import logging
import subprocess
import urllib.parse
import httpx
import numpy as np
import soundfile as sf
from kokoro import KPipeline
from config import Settings
from utils import format_ass_time, sanitize_ass_word, BROWSER_HEADERS, update_live_progress

try:
    from gradio_client import Client as GradioClient, handle_file
    HAS_GRADIO = True
except ImportError:
    HAS_GRADIO = False

logger = logging.getLogger("PAULA.Media")

def run_ffmpeg(cmd: list[str], step_desc: str = "FFmpeg", timeout_sec: int = 180) -> subprocess.CompletedProcess:
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Timeout en {step_desc} tras {timeout_sec}s")
    if res.returncode != 0:
        err_msg = res.stderr.decode("utf-8", errors="ignore")
        tail_err = "\n".join(err_msg.splitlines()[-15:])
        raise RuntimeError(f"Fallo en {step_desc} (Exit {res.returncode}): {tail_err[-250:]}")
    return res

def get_audio_duration(audio_path: str) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path
    ]).decode().strip()
    return float(out)

def _sync_f5_infer(space_id: str, ref_audio: str, script_text: str, hf_token: str) -> str:
    client = GradioClient(space_id, hf_token=hf_token if hf_token else None)
    res = client.predict(
        handle_file(ref_audio),
        "",
        script_text,
        True,
        api_name="/infer"
    )
    out_file = res[0] if isinstance(res, (list, tuple)) else res
    return str(out_file) if out_file else ""

async def generate_voice_hybrid(settings: Settings, script_text: str, audio_path: str, kokoro_pipeline: KPipeline, is_short: bool = True) -> None:
    raw_voice = f"{audio_path}_raw.wav"
    cloned_ok = False

    ref_candidates = [
        f"assets/voices/{settings.channel_id}.wav",
        "assets/voices/default.wav"
    ]
    ref_audio = next((p for p in ref_candidates if os.path.exists(p)), None)

    if HAS_GRADIO and ref_audio:
        for space_id in ["jpgallegoar/Spanish-F5", "mrfakename/E2-F5-TTS"]:
            try:
                logger.info(f"Intentando clonar voz con F5-TTS ({space_id})...")
                out_file = await asyncio.wait_for(
                    asyncio.to_thread(_sync_f5_infer, space_id, ref_audio, script_text, settings.hf_token),
                    timeout=18.0
                )
                if out_file and os.path.exists(out_file):
                    run_ffmpeg([
                        "ffmpeg", "-y", "-i", out_file,
                        "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", raw_voice
                    ], "Conversión F5-TTS")
                    cloned_ok = True
                    break
            except Exception as e:
                logger.warning(f"Salteando F5-TTS ({space_id}) hacia Kokoro local: {e}")

    if not cloned_ok:
        voice_map = {"1": "em_alex", "2": "ef_dora", "3": "em_santa"}
        voice_name = voice_map.get(settings.voice_style, "em_santa")
        speed_val = 1.12 if is_short else 1.04
        logger.info(f"Sintetizando voz neuronal con Kokoro-82M ({voice_name}, speed={speed_val})...")
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

async def prepare_background_music(duration: float, bgm_path: str) -> None:
    bgm_urls = [
        "https://cdn.pixabay.com/download/audio/2022/05/27/audio_1808fbf07a.mp3",
        "https://cdn.pixabay.com/download/audio/2022/01/18/audio_d0a13f69d2.mp3"
    ]
    raw_bgm = f"{bgm_path}_raw.mp3"
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
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

def mix_master_audio_with_sfx(voice_path: str, bgm_path: str, resolved_sfx: list[dict], sfx_bank: dict[str, str], total_duration: float, mixed_audio: str) -> None:
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
        filter_parts.append(f"[{input_idx}:a]volume=0.42,adelay={delay_ms}|{delay_ms}{label}")
        mix_labels.append(label)

    total_inputs = len(mix_labels)
    filter_parts.append(
        f"{''.join(mix_labels)}amix=inputs={total_inputs}:duration=first:dropout_transition=0:normalize=0[aout]"
    )

    run_ffmpeg([
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[aout]", "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", mixed_audio
    ], "Mezcla Multicapa SFX + Voz + BGM")

async def get_word_timestamps(settings: Settings, audio_path: str, script_text: str, total_duration: float) -> list[dict]:
    if settings.groq_api_key and settings.groq_api_key.startswith("gsk_"):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
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
                        headers={
                            "Authorization": f"Bearer {settings.groq_api_key}",
                            "User-Agent": "PAULA-Enterprise-Engine/8.0"
                        },
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
        except Exception as e:
            logger.warning(f"Whisper fallback local activado: {e}")

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
    if any(k in text_sig for k in ["terror", "misteri", "oscuro", "crimen", "miedo", "secreto", "interrogatorio"]):
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

def build_viral_ass_subtitles(words_timed: list[dict], ass_path: str, w: int, h: int, is_short: bool, total_duration: float, channel_name: str = "", topic: str = "") -> None:
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

async def prepare_channel_avatar(settings: Settings, prefix: str) -> str:
    os.makedirs("assets/avatars", exist_ok=True)
    avatar_candidates = [
        f"assets/avatars/{settings.channel_id}.png",
        "assets/avatars/default.png"
    ]
    avatar_img = next((p for p in avatar_candidates if os.path.exists(p)), f"assets/avatars/{settings.channel_id}.png")

    if not os.path.exists(avatar_img):
        prompt = urllib.parse.quote(f"cinematic portrait of charismatic host for {settings.channel_name}, studio lighting, looking at camera, 8k")
        url = f"https://image.pollinations.ai/prompt/{prompt}?width=512&height=512&nologo=true"
        try:
            async with httpx.AsyncClient(timeout=12.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
                r = await client.get(url)
                if r.status_code == 200 and len(r.content) > 5000:
                    with open(avatar_img, "wb") as f:
                        f.write(r.content)
        except Exception:
            pass

    if not os.path.exists(avatar_img):
        return ""

    avatar_vid = f"output/{prefix}_avatar.mp4"
    run_ffmpeg([
        "ffmpeg", "-y", "-loop", "1", "-t", "4.5", "-i", avatar_img,
        "-vf", "scale=360:360,zoompan=z='1.04+0.03*sin(2*PI*on/45)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=135:s=300x300:fps=30,drawbox=x=0:y=0:w=300:h=300:color=0x00FFFF@0.85:t=6",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-threads", "4", avatar_vid
    ], "Render Avatar PiP Local")
    return avatar_vid

def postproducir_video_maestro(input_video: str, output_video: str, is_short: bool = True, trim_silences: bool = True) -> str:
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
        "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-threads", "4",
        "-c:a", "aac", "-b:a", "192k",
        output_video
    ], "Masterización Final")
    return output_video

async def edit_uploaded_raw_video(video_url: str, prefix: str, is_short: bool = True) -> str:
    os.makedirs("output", exist_ok=True)
    raw_input = f"output/{prefix}_uploaded_raw.mp4"
    final_video = f"output/{prefix}_edited_pro.mp4"
    async with httpx.AsyncClient(timeout=90.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
        res = await client.get(video_url)
        res.raise_for_status()
        with open(raw_input, "wb") as f:
            f.write(res.content)
    return postproducir_video_maestro(raw_input, final_video, is_short=is_short, trim_silences=True)

async def build_single_scene_clip(
    settings: Settings,
    i: int,
    term: str,
    prefix: str,
    w: int,
    h: int,
    orientation: str,
    clip_duration: float,
    frames_per_clip: int,
    sem: asyncio.Semaphore
) -> str:
    clip_file = f"output/{prefix}_clip_{i}.mp4"
    vfx_flash = "fade=t=in:st=0:d=0.12,eq=contrast=1.08:saturation=1.15"

    async with sem:
        got_clip = False
        async with httpx.AsyncClient(timeout=18.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
            if settings.visual_style == "1" and settings.pexels_api_key:
                try:
                    pexels_url = f"https://api.pexels.com/videos/search?query={urllib.parse.quote(term)}&orientation={orientation}&per_page=5"
                    r = await client.get(pexels_url, headers={"Authorization": settings.pexels_api_key}, timeout=10.0)
                    videos = r.json().get("videos", [])
                    if videos:
                        chosen = videos[i % len(videos)]
                        vfiles = chosen.get("video_files", [])
                        if vfiles:
                            hd_files = [vf for vf in vfiles if 720 <= vf.get("width", 0) <= 1920] or vfiles
                            best = max(hd_files, key=lambda x: x.get("width", 0))
                            vid_data = await client.get(best["link"], timeout=15.0)
                            raw_vid = f"output/{prefix}_source_{i}.mp4"
                            with open(raw_vid, "wb") as f:
                                f.write(vid_data.content)
                            await asyncio.to_thread(
                                run_ffmpeg,
                                [
                                    "ffmpeg", "-y", "-stream_loop", "-1", "-t", f"{clip_duration:.2f}", "-i", raw_vid,
                                    "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},setsar=1,{vfx_flash},fps=30",
                                    "-r", "30", "-vsync", "cfr", "-video_track_timescale", "15360",
                                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-threads", "2", clip_file
                                ],
                                f"Clip Pexels {i}",
                                90
                            )
                            got_clip = True
                except Exception as e:
                    logger.warning(f"[Scene {i}] Fallback Pexels -> IA: {e}")

            if not got_clip:
                try:
                    gen_w, gen_h = (540, 960) if w < h else (960, 540)
                    ai_prompt = urllib.parse.quote(f"cinematic hyperrealistic scene, {term}, dramatic lighting, 8k, film grain")
                    img_url = f"https://image.pollinations.ai/prompt/{ai_prompt}?width={gen_w}&height={gen_h}&seed={i + 100}&nologo=true"
                    img_res = await client.get(img_url, timeout=16.0)
                    if img_res.status_code == 200 and len(img_res.content) > 4000:
                        scene_img = f"output/{prefix}_scene_{i}.jpg"
                        with open(scene_img, "wb") as f:
                            f.write(img_res.content)
                        z_expr = "min(zoom+0.0018,1.16)" if i % 2 == 0 else "if(eq(on,1),1.16,max(1.0,zoom-0.0018))"
                        await asyncio.to_thread(
                            run_ffmpeg,
                            [
                                "ffmpeg", "-y", "-loop", "1", "-t", f"{clip_duration:.2f}", "-i", scene_img,
                                "-vf", f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},zoompan=z='{z_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames_per_clip}:s={w}x{h}:fps=30,setsar=1,{vfx_flash}",
                                "-r", "30", "-vsync", "cfr", "-video_track_timescale", "15360",
                                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-threads", "2", clip_file
                            ],
                            f"Escena IA {i}",
                            90
                        )
                        got_clip = True
                except Exception as e:
                    logger.warning(f"[Scene {i}] Fallback IA -> Color: {e}")

        if not got_clip:
            await asyncio.to_thread(
                run_ffmpeg,
                [
                    "ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=0x0f172a:s={w}x{h}:d={clip_duration:.2f}:r=30",
                    "-vf", "setsar=1", "-r", "30", "-vsync", "cfr", "-video_track_timescale", "15360",
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", clip_file
                ],
                f"Clip Fallback {i}",
                45
            )

    return os.path.abspath(clip_file).replace("\\", "/")

async def render_video_with_real_clips(
    settings: Settings,
    kokoro_pipeline: KPipeline,
    prefix: str,
    script_text: str,
    search_terms: list[str],
    is_short: bool = True,
    sfx_cues: list[dict] = None
) -> str:
    os.makedirs("output", exist_ok=True)
    w, h = (1080, 1920) if is_short else (1920, 1080)
    orientation = "portrait" if is_short else "landscape"
    format_label = "Short/Reels/TikTok 9:16" if is_short else "Documental 16:9"

    voice_path = f"output/{prefix}_voice.wav"
    bgm_path = f"output/{prefix}_bgm.wav"
    mixed_audio = f"output/{prefix}_master.wav"
    ass_path = f"output/{prefix}_subs.ass"
    raw_output = f"output/{prefix}_raw.mp4"
    final_video = f"output/{prefix}_final.mp4"

    await update_live_progress(settings, 68, f"Generando voz neuronal y alineando fonemas ({format_label})...", "Kokoro-82M / F5-TTS + Whisper Alignment")
    await generate_voice_hybrid(settings, script_text, voice_path, kokoro_pipeline, is_short=is_short)
    total_duration = max(1.0, get_audio_duration(voice_path))

    words_timed = await get_word_timestamps(settings, voice_path, script_text, total_duration)
    build_viral_ass_subtitles(words_timed, ass_path, w, h, is_short, total_duration, settings.channel_name, settings.raw_topic)

    await update_live_progress(settings, 76, f"Sintetizando efectos SFX reales y música ({format_label})...", "Mezcla multicapa sincronizada al milisegundo")
    sfx_bank = ensure_sfx_bank()
    resolved_sfx = resolve_sfx_timestamps(sfx_cues or [], words_timed, total_duration)
    await prepare_background_music(total_duration, bgm_path)
    mix_master_audio_with_sfx(voice_path, bgm_path, resolved_sfx, sfx_bank, total_duration, mixed_audio)

    target_cut_sec = 3.2 if is_short else 4.5
    num_cuts = min(16, max(int(np.ceil(total_duration / target_cut_sec)), len(search_terms), 4))
    clip_duration = total_duration / num_cuts
    frames_per_clip = max(30, int(round(clip_duration * 30)))

    await update_live_progress(settings, 84, f"Renderizando {num_cuts} escenas en paralelo ({format_label})...", "Motor Concurrente + FFmpeg Multithread")

    sem = asyncio.Semaphore(3)
    tasks = [
        build_single_scene_clip(
            settings,
            i,
            search_terms[i % len(search_terms)],
            prefix,
            w,
            h,
            orientation,
            clip_duration,
            frames_per_clip,
            sem
        )
        for i in range(num_cuts)
    ]
    clip_paths = await asyncio.gather(*tasks)

    concat_file = f"output/{prefix}_concat.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for cp in clip_paths:
            f.write(f"file '{cp}'\n")

    await update_live_progress(settings, 92, f"Incrustando Avatar PiP y Subtítulos ASS ({format_label})...", "Masterización final en FFmpeg a 30fps")
    avatar_vid = await prepare_channel_avatar(settings, prefix)
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
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "20", "-threads", "4",
            "-c:a", "aac", "-b:a", "192k", "-shortest", raw_output
        ], "Render con Avatar PiP", 240)
    else:
        run_ffmpeg([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
            "-i", mixed_audio, "-vf", f"ass={ass_filter_path}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "20", "-threads", "4",
            "-c:a", "aac", "-b:a", "192k", "-shortest", raw_output
        ], "Render Estándar", 240)

    postproducir_video_maestro(raw_output, final_video, is_short=is_short, trim_silences=False)
    return final_video
