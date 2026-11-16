import os
import re
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List

logger = logging.getLogger("PAULA.Config")

def clean_secret_token(val: str) -> str:
    """Sanitiza estrictamente tokens y API keys eliminando espacios, comillas y saltos de línea."""
    if not val:
        return ""
    return re.sub(r'[\r\n\t"\'\s]+', '', str(val))

def clean_text_field(val: str) -> str:
    """Limpia textos descriptivos o títulos preservando acentos, eñes y espacios naturales."""
    if not val:
        return ""
    return str(val).strip()

@dataclass
class Settings:
    bot_token: str
    groq_api_key: str
    gemini_api_key: str
    openrouter_api_key: str
    pexels_api_key: str
    hf_token: str
    omnichannel_webhook_url: str
    tiktok_access_token: str
    ig_access_token: str
    x_bearer_token: str
    
    channel_id: str
    channel_name: str
    channel_token_json: str
    
    raw_topic: str
    visual_style: str
    voice_style: str
    raw_count: int
    long_count: int
    short_count: int
    mode_choice: str
    
    publish_at_iso: str
    publish_hour_art: str
    platforms: Dict[str, bool]
    
    monetization_url: str
    custom_title: str
    custom_description: str
    source_video_url: str
    chat_id: str
    status_msg_id: str
    
    gh_owner: str
    gh_repo: str

    @classmethod
    def from_env_or_bundle(cls) -> "Settings":
        raw_bundle = (os.getenv("BUNDLE_JSON") or "").strip()
        bundle: Dict[str, Any] = {}
        if raw_bundle:
            try:
                bundle = json.loads(raw_bundle)
            except Exception as e:
                logger.warning(f"Aviso al parsear BUNDLE_JSON: {e}")
                bundle = {}

        def get_val(env_key: str, bundle_key: str, default: str = "") -> str:
            val = os.getenv(env_key) or bundle.get(bundle_key) or default
            return str(val).strip()

        # Secretos sanitizados con clean_secret_token
        bot_token = clean_secret_token(get_val("TELEGRAM_BOT_TOKEN", "telegram_bot_token"))
        groq_api_key = clean_secret_token(get_val("GROQ_API_KEY", "groq_api_key"))
        gemini_api_key = clean_secret_token(get_val("GEMINI_API_KEY", "gemini_api_key"))
        openrouter_api_key = clean_secret_token(get_val("OPENROUTER_API_KEY", "openrouter_api_key"))
        pexels_api_key = clean_secret_token(get_val("PEXELS_API_KEY", "pexels_api_key"))
        hf_token = clean_secret_token(get_val("HF_TOKEN", "hf_token"))
        
        omnichannel_webhook_url = clean_secret_token(get_val("OMNICHANNEL_WEBHOOK_URL", "omnichannel_webhook_url"))
        tiktok_access_token = clean_secret_token(get_val("TIKTOK_ACCESS_TOKEN", "tiktok_access_token"))
        ig_access_token = clean_secret_token(get_val("IG_ACCESS_TOKEN", "ig_access_token"))
        x_bearer_token = clean_secret_token(get_val("X_BEARER_TOKEN", "x_bearer_token"))
        
        chat_id = clean_secret_token(get_val("CHAT_ID", "chat_id"))
        status_msg_id = clean_secret_token(get_val("STATUS_MSG_ID", "status_msg_id"))

        # Textos y metadatos con clean_text_field (preservan acentos)
        channel_id = clean_text_field(get_val("CHANNEL_ID", "channel_id", "ch1")).lower()
        channel_name = clean_text_field(get_val("CHANNEL_NAME", "channel_name", channel_id.upper()))
        channel_token_json = get_val("CHANNEL_TOKEN_JSON", "channel_token_json", "")
        
        raw_topic = clean_text_field(get_val("TOPIC", "topic", "Los secretos mejor guardados de la tecnología"))
        visual_style = clean_text_field(get_val("VISUAL_STYLE", "visual_style", "2"))
        voice_style = clean_text_field(get_val("VOICE_STYLE", "voice_style", "3"))
        mode_choice = clean_text_field(get_val("MODE", "mode", "3"))
        
        monetization_url = clean_text_field(get_val("MONETIZATION_URL", "monetization_url", ""))
        custom_title = clean_text_field(get_val("CUSTOM_TITLE", "custom_title", ""))
        custom_description = clean_text_field(get_val("CUSTOM_DESCRIPTION", "custom_description", ""))
        source_video_url = clean_text_field(get_val("SOURCE_VIDEO_URL", "source_video_url", ""))
        
        gh_owner = clean_text_field(get_val("GH_OWNER", "gh_owner", "AndreMataCh"))
        gh_repo = clean_text_field(get_val("GH_REPO", "gh_repo", "yt-shorts-factory"))
        
        publish_at_iso = clean_text_field(get_val("PUBLISH_AT_ISO", "publish_at_iso", ""))
        publish_hour_art = clean_text_field(get_val("PUBLISH_HOUR_ART", "publish_hour_art", "NOW"))

        # Validaciones numéricas estrictas con rangos seguros
        raw_count_str = get_val("VIDEO_COUNT", "video_count", "1")
        try:
            raw_count = max(1, min(int(re.sub(r"[^0-9]", "", raw_count_str) or "1"), 10))
        except ValueError:
            raw_count = 1

        raw_long_str = get_val("LONG_COUNT", "long_count", "")
        try:
            if raw_long_str != "":
                long_count = max(0, min(int(re.sub(r"[^0-9]", "", raw_long_str) or "0"), 5))
            else:
                long_count = raw_count if mode_choice in ["2", "3"] else 0
        except ValueError:
            long_count = 1

        raw_short_str = get_val("SHORT_COUNT", "short_count", "")
        try:
            if raw_short_str != "":
                short_count = max(0, min(int(re.sub(r"[^0-9]", "", raw_short_str) or "0"), 10))
            else:
                short_count = raw_count if mode_choice in ["1", "3"] else 0
        except ValueError:
            short_count = 1

        if long_count == 0 and short_count == 0:
            short_count = 1

        platforms_raw = bundle.get("platforms")
        if isinstance(platforms_raw, dict):
            platforms = {str(k): bool(v) for k, v in platforms_raw.items()}
        else:
            platforms = {"youtube": True, "tiktok": True, "reels": True, "x": True}

        return cls(
            bot_token=bot_token,
            groq_api_key=groq_api_key,
            gemini_api_key=gemini_api_key,
            openrouter_api_key=openrouter_api_key,
            pexels_api_key=pexels_api_key,
            hf_token=hf_token,
            omnichannel_webhook_url=omnichannel_webhook_url,
            tiktok_access_token=tiktok_access_token,
            ig_access_token=ig_access_token,
            x_bearer_token=x_bearer_token,
            channel_id=channel_id,
            channel_name=channel_name,
            channel_token_json=channel_token_json,
            raw_topic=raw_topic,
            visual_style=visual_style,
            voice_style=voice_style,
            raw_count=raw_count,
            long_count=long_count,
            short_count=short_count,
            mode_choice=mode_choice,
            publish_at_iso=publish_at_iso,
            publish_hour_art=publish_hour_art,
            platforms=platforms,
            monetization_url=monetization_url,
            custom_title=custom_title,
            custom_description=custom_description,
            source_video_url=source_video_url,
            chat_id=chat_id,
            status_msg_id=status_msg_id,
            gh_owner=gh_owner,
            gh_repo=gh_repo
        )
