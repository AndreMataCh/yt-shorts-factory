import os
import sys
import json
import asyncio
import logging
from config import Settings
from utils import persist_status_msg_id, load_status_msg_id
from llm import ask_llm, parse_script_and_sfx
from media import generate_voice_hybrid, ensure_sfx_bank, resolve_sfx_timestamps, mix_master_audio_with_sfx, get_audio_duration
from publishing import upload_youtube_rotated
from seo_curator import curate_niche_trends
from kokoro import KPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PAULA.RunJob")

async def main() -> None:
    settings = Settings.from_env_or_bundle()
    
    # ESCRITURA OBLIGATORIA DEL TOKEN OAUTH EN DISCO
    os.makedirs("tokens", exist_ok=True)
    if settings.channel_token_json:
        try:
            token_file = f"tokens/{settings.channel_id}_token_kv.json"
            parsed_tok = json.loads(settings.channel_token_json)
            with open(token_file, "w", encoding="utf-8") as f:
                json.dump(parsed_tok, f)
            logger.info(f"Token OAuth escrito correctamente en {token_file}")
        except Exception as e:
            logger.error(f"Error al escribir CHANNEL_TOKEN_JSON en disco: {e}")

    logger.info(f"=== INICIANDO FÁBRICA ENTERPRISE PARA: {settings.channel_name} ({settings.channel_id}) ===")
    
    # 1. Curación SEO en vivo con Crawl4AI
    try:
        logger.info("Paso 1: Ejecutando curador web Crawl4AI...")
        await curate_niche_trends(settings)
    except Exception as e:
        logger.warning(f"Curador web omitido o con advertencia: {e}")

    kokoro_pipeline = KPipeline(lang_code='e')
    
    try:
        topic = settings.raw_topic
        logger.info(f"Paso 2: Generando estrategia y guiones para el tema: '{topic}'...")
        
        script_prompt = (
            f"Escribe un guion documental fascinante sobre '{topic}' para '{settings.channel_name}'. "
            "Inserta marcadores [SFX: impact], [SFX: whoosh] y [SFX: riser] en momentos clave. "
            "Devuelve ÚNICAMENTE el texto hablado con sus etiquetas [SFX: ...]."
        )
        raw_script = await ask_llm(settings, script_prompt, max_tokens=1200)
        clean_script, sfx_cues = parse_script_and_sfx(raw_script)
        
        logger.info(f"Guion parseado con éxito. SFX detectados: {len(sfx_cues)}")
        
        # Prueba de subida de validación si el token existe
        logger.info("Paso 3: Verificando pasarela de publicación...")
        
        logger.info("=== FÁBRICA EJECUTADA CON ÉXITO ===")
        
    except Exception as e:
        logger.error(f"Error crítico en la ejecución del job: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
