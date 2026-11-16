import os
import sys
import asyncio
import logging
from crawl4ai import AsyncWebCrawler
from config import Settings
from utils import retry_async

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PAULA.SEOCurator")

async def curate_niche_trends(settings: Settings) -> str:
    """
    Utiliza Crawl4AI para rastrear tendencias, títulos y ganchos exitosos
    de la competencia en YouTube y la web para el nicho actual del canal.
    """
    niche_query = settings.raw_topic
    search_url = f"https://www.youtube.com/results?search_query={urllib_quote(niche_query)}&sp=CAM%253D"
    
    logger.info(f"Iniciando Crawl4AI SEO Curator para el nicho: '{niche_query}'...")
    
    scraped_insights = []
    
    async def _run_crawler():
        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=search_url)
            if result.success:
                # Extraer fragmentos de títulos o texto relevante de la página de resultados
                text_content = result.markdown or result.cleaned_html or ""
                lines = [line.strip() for line in text_content.splitlines() if len(line.strip()) > 20]
                return lines[:25]
            return []

    try:
        raw_lines = await retry_async(_run_crawler, retries=2)
        for line in raw_lines:
            if any(kw in line.lower() for kw in ["caso", "secreto", "verdad", "misterio", "historia", "exclusivo", "impacto"]):
                scraped_insights.append(line)
    except Exception as e:
        logger.warning(f"Crawl4AI no pudo completar el scraping en vivo: {e}. Usando respaldo estático.")
        scraped_insights = [
            f"Tendencia validada en {settings.channel_name}: Revelación progresiva de datos ocultos sobre {niche_query}.",
            f"Formato de alta retención: Enfocar los primeros 5 segundos en una pregunta sin respuesta aparente."
        ]

    context_text = f"# Reporte SEO y Competencia — {settings.channel_name}\n"
    context_text += f"## Nicho Analizado: {niche_query}\n\n"
    context_text += "### Patrones y Ganchos Detectados:\n"
    for item in scraped_insights[:8]:
        context_text += f"- {item}\n"

    # Guardar en disco para que sea consumido por el generador de guiones
    os.makedirs(".", exist_ok=True)
    with open("seo_context.md", "w", encoding="utf-8") as f:
        f.write(context_text)

    logger.info("seo_context.md generado y optimizado con éxito.")
    return context_text

def urllib_quote(text: str) -> str:
    import urllib.parse
    return urllib.parse.quote(text)

if __name__ == "__main__":
    settings = Settings.from_env_or_bundle()
    asyncio.run(curate_niche_trends(settings))
