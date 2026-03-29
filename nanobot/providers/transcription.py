"""Voice transcription provider using Groq."""

import os
from io import BufferedReader, _BufferedReaderStream
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
                    files: dict[str, tuple[str, BufferedReader[_BufferedReaderStream]] | tuple[None, str]] = {
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

    def __init__(self, llm_provider_config: ProviderConfig):
        self.llm_provider_config = llm_provider_config
    
    async def transcribe(self, file_path: str | Path) -> str:
        path: Path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        try:
            async with httpx.AsyncClient() as client:
                config = self.llm_provider_config
                with open(path, "rb") as f:
                    files: dict[str, tuple[str, BufferedReader[_BufferedReaderStream]] | tuple[None, str]] = {
                        "file": (path.name, f),
                    }

                    if config.extra_headers and "model" in config.extra_headers:
                        files["model"] = (None, str(config.extra_headers["model"]))
                    
                    if config.api_key:
                        headers = {
                            "Authorization": f"Bearer {config.api_key}"
                        }

                    response = await client.post(
                        config.api_base + "/audio/transcriptions",
                        headers=headers,
                        files=files,
                        data={"language": config.extra_headers.get("language", "en")},
                        timeout=60.0
                    )

                    response.raise_for_status()
                    data = response.json()
                    return data.get("text", "")

        except Exception as e:
            logger.error("Audio provider transcription error: {}", e)
            return ""

