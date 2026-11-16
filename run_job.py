import os
import sys
import asyncio
import logging
from config import Settings
from utils import persist_status_msg_id, load_status_msg_id
from llm import ask_llm, parse_script_and_sfx
from media import generate_voice_hybrid, ensure_sfx_bank, resolve_sfx_timestamps, mix_master_audio_with_sfx, get_audio_duration
from publishing import upload_youtube_rotated
from kokoro import KPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PAULA.Main")

async def main() -> None:
    settings = Settings.from_env_or_bundle()
    status_file = "output/.status_msg_id.json"
    status_msg_id = load_status_msg_id(status_file)

    logger.info(f"Iniciando fábrica Omnicanal para canal: {settings.channel_name}")
    
    kokoro_pipeline = KPipeline(lang_code='e')
    
    try:
        topic = settings.raw_topic
        long_plan = await generate_long_plan_modular(settings, topic, kokoro_pipeline)
        
        # Ejemplo de flujo para un video largo
        logger.info("Renderizando video largo...")
        # (Aquí se conecta con las funciones de render optimizadas en media.py)
        
        logger.info("¡Tanda completada con éxito!")
    except Exception as e:
        logger.error(f"Error crítico en ejecución: {e}")
        raise e

async def generate_long_plan_modular(settings: Settings, topic: str, kokoro_pipeline: KPipeline) -> dict:
    script_prompt = (
        f"Escribe un guion documental fascinante sobre '{topic}' para '{settings.channel_name}'. "
        "Inserta marcadores [SFX: impact] y [SFX: whoosh] en momentos clave. "
        "Devuelve ÚNICAMENTE el texto hablado con sus etiquetas [SFX: ...]."
    )
    raw_script = await ask_llm(settings, script_prompt, max_tokens=1200)
    clean_script, sfx_cues = parse_script_and_sfx(raw_script)
    
    title = settings.custom_title or f"Caso Definitivo: {topic[:50]}"
    return {"title": title, "script": clean_script, "sfx_cues": sfx_cues, "search_terms": [topic]}

if __name__ == "__main__":
    asyncio.run(main())
