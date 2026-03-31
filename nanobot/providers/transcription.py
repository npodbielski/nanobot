"""Voice transcription provider using Groq."""

import os
from pathlib import Path

import httpx
from loguru import logger

from nanobot.config.schema import ProviderConfig


class GroqTranscriptionProvider:
    """
    Voice transcription provider using Groq's Whisper API.

    Groq offers extremely fast transcription with a generous free tier.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using Groq.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text.
        """
        if not self.api_key:
            logger.warning("Groq API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {
                        "file": (path.name, f),
                        "model": (None, "whisper-large-v3"),
                    }
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                    }

                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        files=files,
                        timeout=60.0
                    )

                    response.raise_for_status()
                    data = response.json()
                    return data.get("text", "")

        except Exception as e:
            logger.error("Groq transcription error: {}", e)
            return ""


class TranscriptionProvider:
    """Initialize the transcription provider with the necessary configuration.

    :param llm_provider_config: Configuration settings for the LLM provider.
    """

    def __init__(self, llm_provider_config: ProviderConfig):
        self.llm_provider_config = llm_provider_config

    async def transcribe(self, file_path: str | Path) -> str:
        """Transcribe audio file content to text using the configured LLM provider.

            :param file_path: The path to the audio file to be transcribed.
            :return: The transcribed text content, or an empty string on failure.
        """
        path: Path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""
        try:
            async with httpx.AsyncClient() as client:
                config = self.llm_provider_config
                with open(path, "rb") as f:
                    files = {
                        "file": (path.name, f),
                    }

                    if config.extra_headers and "model" in config.extra_headers:
                        files["model"] = (None, str(config.extra_headers["model"]))

                    h: dict[str, str] = {}
                    if "api_key" in config:
                        h = {
                            "Authorization": f"Bearer {config.api_key}"
                        }

                    data = None
                    if config.extra_headers and "language" in config.extra_headers:
                        data = {"language": config.extra_headers["language"]}

                    response = await client.post(
                        config.api_base + "/audio/transcriptions",
                        headers=h,
                        files=files,
                        data=data,
                        timeout=60.0
                    )

                    response.raise_for_status()
                    data = response.json()
                    return data.get("text", "")

        except Exception as e:
            logger.error("Custom provider transcription error: {}", e)
            return ""
