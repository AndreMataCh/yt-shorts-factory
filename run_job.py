import os
import sys
import asyncio
import warnings
import logging

os.environ["PYTHONUNBUFFERED"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

from config import Settings

# Cargar configuración e inyectar HF_TOKEN antes de importar Kokoro
_boot_settings = Settings.from_env_or_bundle()
if _boot_settings.hf_token:
    os.environ["HF_TOKEN"] = _boot_settings.hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _boot_settings.hf_token

from kokoro import KPipeline
from utils import update_live_progress, send_telegram, persist_status_msg_id, escape_html
from llm import ask_llm, generate_long_plan, generate_teaser_short_plan
from media import render_video_with_real_clips, edit_uploaded_raw_video
from publishing import upload_youtube_rotated, distribute_omnichannel_short
from seo_curator import curate_niche_trends

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PAULA.RunJob")

async def main() -> None:
    settings = _boot_settings
    os.makedirs("output", exist_ok=True)
    os.makedirs("tokens", exist_ok=True)

    if settings.status_msg_id:
        persist_status_msg_id("output/.status_msg_id.json", settings.status_msg_id)

    logger.info(
        f"=== INICIANDO FÁBRICA ENTERPRISE PARA: {settings.channel_name} ({settings.channel_id}) | "
        f"{settings.long_count} Largos + {settings.short_count} Shorts ==="
    )

    try:
        await update_live_progress(
            settings,
            50,
            "Auditando tendencias de la competencia con Crawl4AI...",
            f"Canal: {settings.channel_name} | Tema: {settings.raw_topic}"
        )
        try:
            await curate_niche_trends(settings)
        except Exception as e:
            logger.warning(f"Curador web omitido: {e}")

        logger.info("Inicializando motor neuronal Kokoro-82M...")
        kokoro_pipeline = KPipeline(lang_code='e')

        if settings.source_video_url:
            is_short_mode = (settings.short_count > 0 and settings.long_count == 0)
            edited_vid = await edit_uploaded_raw_video(settings.source_video_url, "raw_edit", is_short=is_short_mode)
            title = settings.custom_title or settings.raw_topic[:80]
            desc = settings.custom_description or f"{title}\n\n{settings.monetization_url}"
            vid_url = await upload_youtube_rotated(settings, edited_vid, title, desc, ["viral", "pro"], is_short=is_short_mode)
            await update_live_progress(settings, 100, "¡Video Pregrabado Publicado! ✅", vid_url)
            await send_telegram(settings, f"✅ <b>Video Pregrabado Masterizado y Publicado:</b>\n🔗 {vid_url}")
            return

        max_needed = max(settings.long_count, settings.short_count, 1)
        if max_needed > 1:
            variation_prompt = (
                f"Para el tema '{settings.raw_topic}', genera {max_needed} ángulos virales distintos. "
                "Devuelve ÚNICAMENTE los títulos separados por punto y coma (;)."
            )
            raw_subtopics = await ask_llm(settings, variation_prompt, max_tokens=350)
            subtopics = [s.strip() for s in raw_subtopics.split(";") if s.strip()][:max_needed]
            while len(subtopics) < max_needed:
                subtopics.append(f"{settings.raw_topic} - Parte {len(subtopics) + 1}")
        else:
            subtopics = [settings.raw_topic]

        published_summary = []
        primary_long_url = ""
        primary_long_title = settings.raw_topic

        # 1. PRODUCCIÓN Y SUBIDA DE VIDEOS LARGOS (16:9)
        for idx in range(1, settings.long_count + 1):
            topic = subtopics[(idx - 1) % len(subtopics)]
            await update_live_progress(settings, 60, f"Diseñando Video Largo ({idx}/{settings.long_count})...", topic)
            long_plan = await generate_long_plan(settings, topic)
            long_vid = await render_video_with_real_clips(
                settings,
                kokoro_pipeline,
                f"long_{idx}",
                long_plan["script"],
                long_plan["search_terms"],
                is_short=False,
                sfx_cues=long_plan.get("sfx_cues", [])
            )
            if settings.platforms.get("youtube", True):
                await update_live_progress(settings, 94, f"Subiendo Video Largo ({idx}/{settings.long_count}) a YouTube...", long_plan["title"])
                long_url = await upload_youtube_rotated(
                    settings,
                    long_vid,
                    long_plan["title"],
                    long_plan["description"],
                    long_plan["tags"],
                    is_short=False
                )
                if not primary_long_url:
                    primary_long_url = long_url
                    primary_long_title = long_plan["title"]
                sched_badge = f" <i>(Prog: {settings.publish_hour_art} ART)</i>" if settings.publish_at_iso and settings.publish_hour_art != "NOW" else ""
                published_summary.append(f"🖥️ <b>Largo #{idx}:</b> <a href='{long_url}'>{escape_html(long_plan['title'])}</a>{sched_badge}")

        # 2. PRODUCCIÓN Y SUBIDA DE SHORTS / REELS / TIKTOKS (9:16)
        for idx in range(1, settings.short_count + 1):
            topic = subtopics[(idx - 1) % len(subtopics)]
            await update_live_progress(settings, 66, f"Diseñando Short / Reel / TikTok ({idx}/{settings.short_count})...", topic)
            short_plan = await generate_teaser_short_plan(settings, topic, primary_long_title, primary_long_url)
            short_vid = await render_video_with_real_clips(
                settings,
                kokoro_pipeline,
                f"short_{idx}",
                short_plan["script"],
                short_plan["search_terms"],
                is_short=True,
                sfx_cues=short_plan.get("sfx_cues", [])
            )
            short_url = ""
            if settings.platforms.get("youtube", True):
                await update_live_progress(settings, 97, f"Subiendo Short ({idx}/{settings.short_count}) a YouTube...", short_plan["title"])
                short_url = await upload_youtube_rotated(
                    settings,
                    short_vid,
                    short_plan["title"],
                    short_plan["description"],
                    short_plan["tags"],
                    is_short=True
                )
                sched_badge = f" <i>(Prog: {settings.publish_hour_art} ART)</i>" if settings.publish_at_iso and settings.publish_hour_art != "NOW" else ""
                published_summary.append(f"📱 <b>Short #{idx}:</b> <a href='{short_url}'>{escape_html(short_plan['title'])}</a>{sched_badge}")

            omni_notes = await distribute_omnichannel_short(settings, short_plan, short_vid, short_url)
            published_summary.extend(omni_notes)

        links_block = "\n".join(published_summary) if published_summary else "<i>Sin activos marcados para subida.</i>"
        gum_line = f"\n\n💰 <b>Producto Digital / Gumroad:</b> {settings.monetization_url}" if settings.monetization_url else ""
        await update_live_progress(settings, 100, "¡Producción Omnicanal Completada! 🏁", f"Activos publicados en {settings.channel_name}.")
        await send_telegram(
            settings,
            f"🏁 <b>¡Tanda Omnicanal Finalizada en {escape_html(settings.channel_name)}!</b>\n"
            f"🕒 <b>Programación:</b> {escape_html(settings.publish_hour_art)} ART\n\n"
            f"{links_block}{gum_line}"
        )
        logger.info("=== FÁBRICA EJECUTADA Y PUBLICADA CON ÉXITO ===")

    except Exception as e:
        logger.error(f"Error crítico en la ejecución del job: {e}")
        await update_live_progress(settings, 100, "❌ Error durante la ejecución", str(e)[:180])
        await send_telegram(
            settings,
            f"❌ <b>Error crítico en fábrica ({escape_html(settings.channel_name)}):</b>\n<code>{escape_html(str(e)[:350])}</code>"
        )
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
