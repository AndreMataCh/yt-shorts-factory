import os
import re
import json
import asyncio
import urllib.parse
import httpx

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    HAS_CRAWL4AI = True
except ImportError:
    HAS_CRAWL4AI = False

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
    except Exception:
        BUNDLE = {}

def get_cfg(env_key: str, bundle_key: str, default: str = "") -> str:
    val = os.getenv(env_key) or BUNDLE.get(bundle_key) or default
    return str(val).strip()

TOPIC = get_cfg("TOPIC", "topic", "Tecnología y Misterios")
CHANNEL_NAME = get_cfg("CHANNEL_NAME", "channel_name", "Canal Principal")
GEMINI_API_KEY = clean_secret(get_cfg("GEMINI_API_KEY", "gemini_api_key"))
GROQ_API_KEY = clean_secret(get_cfg("GROQ_API_KEY", "groq_api_key"))
OPENROUTER_API_KEY = clean_secret(get_cfg("OPENROUTER_API_KEY", "openrouter_api_key"))

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
}

async def scrape_live_trends(query: str) -> str:
    encoded_q = urllib.parse.quote(query)
    urls = [
        f"https://html.duckduckgo.com/html/?q={encoded_q}",
        f"https://news.google.com/rss/search?q={encoded_q}&hl=es-419&gl=AR&ceid=AR:es-419"
    ]
    extracted_chunks = []

    if HAS_CRAWL4AI:
        try:
            print(f"🕷️ Crawl4AI: Analizando tendencias web para '{query}'...")
            browser_cfg = BrowserConfig(headless=True, verbose=False)
            run_cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                for u in urls:
                    try:
                        res = await crawler.arun(url=u, config=run_cfg)
                        if res and res.markdown:
                            extracted_chunks.append(res.markdown[:3500])
                    except Exception as e:
                        print(f"Aviso Crawl4AI en {u}: {e}")
        except Exception as e:
            print(f"Aviso inicializando Crawl4AI: {e}")

    # Respaldo directo HTTPX si Crawl4AI no extrajo texto suficiente
    if not extracted_chunks:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers=BROWSER_HEADERS) as client:
            for u in urls:
                try:
                    r = await client.get(u)
                    if r.status_code == 200:
                        text_clean = re.sub(r"<[^>]+>", " ", r.text)
                        text_clean = re.sub(r"\s+", " ", text_clean).strip()
                        extracted_chunks.append(text_clean[:3500])
                except Exception:
                    pass

    return "\n\n---\n\n".join(extracted_chunks)[:6500]

async def synthesize_seo_brief(topic: str, raw_web_data: str) -> str:
    prompt = (
        f"Eres un estratega SEO de YouTube para el canal '{CHANNEL_NAME}'.\n"
        f"Tema a producir: '{topic}'.\n\n"
        f"Datos extraídos de la web en tiempo real:\n{raw_web_data[:4500]}\n\n"
        "Genera un informe conciso en Markdown (máximo 250 palabras) que incluya:\n"
        "1. Datos curiosos reales, cifras o revelaciones específicas para enriquecer el guion.\n"
        "2. 5 palabras clave (keywords) de alto volumen de búsqueda.\n"
        "3. Un ángulo psicológico o gancho de retención para los primeros 5 segundos."
    )

    if GEMINI_API_KEY and GEMINI_API_KEY.startswith("AIza"):
        headers = {**BROWSER_HEADERS, "x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=40.0, headers=headers) as client:
            for model in ["gemini-2.5-flash", "gemini-2.0-flash"]:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                    res = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
                    if res.status_code == 200:
                        cands = res.json().get("candidates", [])
                        if cands:
                            txt = cands[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
                            if txt:
                                return txt
                except Exception:
                    pass

    if GROQ_API_KEY and GROQ_API_KEY.startswith("gsk_"):
        headers = {**BROWSER_HEADERS, "Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=40.0, headers=headers) as client:
            try:
                res = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.6}
                )
                if res.status_code == 200:
                    txt = res.json()["choices"][0]["message"].get("content", "").strip()
                    if txt:
                        return txt
            except Exception:
                pass

    if OPENROUTER_API_KEY:
        headers = {**BROWSER_HEADERS, "Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=40.0, headers=headers) as client:
            try:
                res = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json={"model": "meta-llama/llama-3.3-70b-instruct:free", "messages": [{"role": "user", "content": prompt}]}
                )
                if res.status_code == 200:
                    txt = res.json()["choices"][0]["message"].get("content", "").strip()
                    if txt:
                        return txt
            except Exception:
                pass

    async with httpx.AsyncClient(timeout=40.0, headers={**BROWSER_HEADERS, "Content-Type": "application/json"}, follow_redirects=True) as client:
        try:
            res = await client.post(
                "https://text.pollinations.ai/openai",
                json={"model": "openai", "messages": [{"role": "user", "content": prompt}]}
            )
            if res.status_code == 200:
                txt = res.json()["choices"][0]["message"].get("content", "").strip()
                if txt:
                    return txt
        except Exception:
            pass

    return f"Enfoque viral recomendado para {topic}: destacar datos verificables, ritmo documental ágil y misterio progresivo."

async def main():
    print(f"🔍 Ejecutando SEO Curator (Crawl4AI) para: {TOPIC}")
    raw_web = await scrape_live_trends(TOPIC)
    brief = await synthesize_seo_brief(TOPIC, raw_web)
    with open("seo_context.md", "w", encoding="utf-8") as f:
        f.write(brief)
    print("✅ Archivo seo_context.md generado con éxito.")

if __name__ == "__main__":
    asyncio.run(main())
