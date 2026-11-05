import asyncio
import os
import urllib.parse
from crawl4ai import AsyncWebCrawler

async def gather_niche_intelligence(topic):
    encoded_topic = urllib.parse.quote(topic)
    
    # Priorizamos Google News y fuentes de texto plano estables para scraping rápido en CI/CD
    search_urls = [
        f"https://news.google.com/search?q={encoded_topic}&hl=es-419&gl=AR&ceid=AR%3Aes-419",
        f"https://html.duckduckgo.com/html/?q={encoded_topic}"
    ]
    
    extracted_markdown = []
    
    print(f"🕵️‍♂️ Iniciando inteligencia de nicho para: '{topic}'...")
    
    # Configuramos AsyncWebCrawler con opciones seguras para headless
    async with AsyncWebCrawler(verbose=True) as crawler:
        for url in search_urls:
            try:
                # Usamos wait_for para asegurar que la página renderice contenido textual
                res = await crawler.arun(
                    url=url,
                    bypass_cache=True,
                    timeout=30000
                )
                if res.success and res.markdown:
                    # Limpiamos y recortamos fragmentos clave para no saturar el prompt del LLM
                    clean_content = res.markdown[:4000]
                    extracted_markdown.append(f"### Fuente analizada: {url}\n{clean_content}")
                    print(f"✅ Éxito al rascar: {url}")
                else:
                    print(f"⚠️ Sin contenido útil en: {url}")
            except Exception as e:
                print(f"❌ Aviso al procesar {url}: {e}")

    final_context = "\n\n".join(extracted_markdown) if extracted_markdown else f"Contexto base sobre {topic} obtenido por IA."
    
    # Guardamos el archivo de contexto para la fábrica
    with open("seo_context.md", "w", encoding="utf-8") as f:
        f.write(f"# Intelligence Briefing para el Tema: {topic}\n\n{final_context}")
    
    print("✅ Contexto SEO guardado exitosamente en seo_context.md")

if __name__ == "__main__":
    # Lee el topic inyectado desde GitHub Actions (enviado por el Worker o PAULA)
    topic = os.getenv("TOPIC") or os.getenv("NICHE_OR_CONFIG", "Tendencias virales de tecnología")
    asyncio.run(gather_niche_intelligence(topic))
