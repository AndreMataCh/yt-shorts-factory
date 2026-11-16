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
from utils import format_ass_time, BROWSER_HEADERS

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

async def generate_voice_hybrid(settings: Settings, script_text: str, audio_path: str, kokoro_pipeline: KPipeline, is_short: bool = True) -> None:
    raw_voice = f"{audio_path}_raw.wav"
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
    """Genera físicamente los archivos WAV de efectos de sonido (SFX) si no existen en disco."""
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

def mix_master_audio_with_sfx(voice_path: str, bgm_path: str, resolved_sfx: list[dict], sfx_bank: dict[str, str], total_duration: float, mixed_audio: str) -> None:
    """Mezcla multicanal inyectando los SFX generados exactamente en los milisegundos indicados."""
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
