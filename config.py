import os
import json
import logging

logger = logging.getLogger("PAULA.Config")


def _parse_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on", "si", "sí")


def _parse_int(value, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


class Settings:
    """Configuración centralizada de PAULA v8.0 deserializada desde BUNDLE_JSON y entorno."""

    def __init__(self, bundle: dict | None = None):
        data = bundle if isinstance(bundle, dict) else {}

        # Identificación de canal y ejecución
        self.channel_id: str = str(data.get("channel_id") or os.getenv("CHANNEL_ID") or "ch1").strip()
        self.channel_name: str = str(data.get("channel_name") or os.getenv("CHANNEL_NAME") or "Canal Omnicanal").strip()
        self.raw_topic: str = str(data.get("topic") or os.getenv("TOPIC") or "Documental y análisis de alto impacto").strip()
        self.custom_title: str = str(data.get("custom_title") or "").strip()
        self.custom_description: str = str(data.get("custom_description") or "").strip()
        self.mode_choice: str = str(data.get("mode") or os.getenv("MODE") or "3").strip()
        self.chat_id: str = str(data.get("chat_id") or os.getenv("CHAT_ID") or "").strip()
        self.status_msg_id: str = str(data.get("status_msg_id") or "").strip()

        # Estilos multimedia y cuotas de tanda
        self.visual_style: str = str(data.get("visual_style") or os.getenv("VISUAL_STYLE") or "2").strip()
        self.voice_style: str = str(data.get("voice_style") or os.getenv("VOICE_STYLE") or "3").strip()
        self.long_count: int = max(0, _parse_int(data.get("long_count", os.getenv("LONG_COUNT", 1)), 1))
        self.short_count: int = max(0, _parse_int(data.get("short_count", os.getenv("SHORT_COUNT", 2)), 2))
        self.video_count: int = max(
            1,
            _parse_int(data.get("video_count", os.getenv("VIDEO_COUNT", max(self.long_count, self.short_count, 1))), 1),
        )

        # Programación horaria (ART / ISO) y monetización
        self.publish_hour_art: str = str(data.get("publish_hour_art") or os.getenv("PUBLISH_HOUR_ART") or "20:00").strip()
        self.optimal_publish_hour_art: str = str(
            data.get("optimal_publish_hour_art") or self.publish_hour_art or "20:00"
        ).strip()
        self.publish_at_iso: str = str(data.get("publish_at_iso") or os.getenv("PUBLISH_AT_ISO") or "").strip()
        self.monetization_url: str = str(data.get("monetization_url") or "").strip()

        # Activos personalizados: logo, intro y outro (videos largos 16:9)
        self.custom_logo_url: str = str(data.get("custom_logo_url") or os.getenv("CUSTOM_LOGO_URL") or "").strip()
        self.intro_url: str = str(data.get("intro_url") or os.getenv("INTRO_URL") or "").strip()
        self.outro_url: str = str(data.get("outro_url") or os.getenv("OUTRO_URL") or "").strip()

        # Credenciales OAuth y APIs
        raw_token = data.get("channel_token_json", os.getenv("CHANNEL_TOKEN_JSON", ""))
        self.channel_token_json: str = (
            json.dumps(raw_token) if isinstance(raw_token, dict) else str(raw_token or "").strip()
        )
        self.google_client_id: str = str(
            data.get("google_client_id") or os.getenv("GOOGLE_CLIENT_ID") or os.getenv("YOUTUBE_CLIENT_ID") or ""
        ).strip()
        self.google_client_secret: str = str(
            data.get("google_client_secret") or os.getenv("GOOGLE_CLIENT_SECRET") or os.getenv("YOUTUBE_CLIENT_SECRET") or ""
        ).strip()

        self.gemini_api_key: str = str(data.get("gemini_api_key") or os.getenv("GEMINI_API_KEY") or "").strip()
        self.groq_api_key: str = str(data.get("groq_api_key") or os.getenv("GROQ_API_KEY") or "").strip()
        self.openrouter_api_key: str = str(data.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY") or "").strip()
        self.pexels_api_key: str = str(data.get("pexels_api_key") or os.getenv("PEXELS_API_KEY") or "").strip()
        self.hf_token: str = str(data.get("hf_token") or os.getenv("HF_TOKEN") or "").strip()
        self.bot_token: str = str(data.get("telegram_bot_token") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
        self.omnichannel_webhook_url: str = str(
            data.get("omnichannel_webhook_url") or os.getenv("OMNICHANNEL_WEBHOOK_URL") or ""
        ).strip()
        self.tiktok_access_token: str = str(data.get("tiktok_access_token") or os.getenv("TIKTOK_ACCESS_TOKEN") or "").strip()
        self.ig_access_token: str = str(data.get("ig_access_token") or os.getenv("IG_ACCESS_TOKEN") or "").strip()
        self.x_bearer_token: str = str(data.get("x_bearer_token") or os.getenv("X_BEARER_TOKEN") or "").strip()

        # Plataformas destino
        plats = data.get("platforms")
        if not isinstance(plats, dict):
            plats = {}
        self.platforms: dict[str, bool] = {
            "youtube": _parse_bool(plats.get("youtube"), True),
            "tiktok": _parse_bool(plats.get("tiktok"), True),
            "reels": _parse_bool(plats.get("reels"), True),
            "x": _parse_bool(plats.get("x"), True),
        }

    @classmethod
    def from_env_or_bundle(cls) -> "Settings":
        raw_bundle = os.getenv("BUNDLE_JSON", "").strip()
        if not raw_bundle:
            return cls({})
        try:
            parsed = json.loads(raw_bundle)
            if isinstance(parsed, dict):
                return cls(parsed)
        except Exception as exc:
            logger.warning(f"No se pudo parsear BUNDLE_JSON ({exc}). Usando valores por defecto/entorno.")
        return cls({})
