from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr

from .log import logger
from .models import TTSEncoding, TTSLanguages, TTSModels

NUM_CHANNELS = 1
SENTENCE_END_REGEX = re.compile(r'.*[-.—!?,;:…।|]$')
API_BASE_URL = "https://waves-api.smallest.ai/api/v1"


@dataclass
class _TTSOptions:
    model: TTSModels
    encoding: TTSEncoding
    voice: str
    api_key: str
    language: TTSLanguages # en, hi, mr, kn, ta, bn, gu, de, fr, es, it, pl, nl, ru, ar, he
    sample_rate: NotGivenOr[int] = NOT_GIVEN  # 8000-24000, default 24000
    add_wav_header: NotGivenOr[bool] = NOT_GIVEN # default False
    transliterate: NotGivenOr[bool] = NOT_GIVEN # default False
    similarity: NotGivenOr[float] = NOT_GIVEN  # 0-1, default 0
    speed: NotGivenOr[float] = NOT_GIVEN  # 0.5-2.0, default 1.0
    consistency: NotGivenOr[float] = NOT_GIVEN  # 0-1, default 0.5
    enhancement: NotGivenOr[int] = NOT_GIVEN  # 0-2, default 1


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        model: TTSModels | str = "lightning-v2",
        sample_rate: NotGivenOr[int] = 24000,
        language: str = "en",
        voice: str = "emily",
        add_wav_header: NotGivenOr[bool] = False,
        transliterate: NotGivenOr[bool] = False,
        similarity: NotGivenOr[float] = 0,
        speed: NotGivenOr[float] = 1.0,
        consistency: NotGivenOr[float] = 0.5,
        enhancement: NotGivenOr[int] = 1,
        encoding: TTSEncoding = "pcm_s16le",
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """
        Create a new instance of smallest.ai Waves TTS.
        Args:
            model (TTSModels, optional): The Waves TTS model to use. Defaults to "lightning".
            language (TTSLanguages, optional): The language code for synthesis. Defaults to "en".
            encoding (TTSEncoding, optional): The audio encoding format. Defaults to "pcm_s16le".
            voice (VoiceSettings, optional): The voice settings to use. Defaults to "emily".
            sample_rate (int, optional): The audio sample rate in Hz. Defaults to 24000.
            api_key (str, optional): The smallest.ai API key. If not provided, it will be read from the SMALLEST_API_KEY environment variable.
            add_wav_header (bool, optional): If True, includes a WAV header in the audio output; otherwise, only raw audio data is returned.
            http_session (aiohttp.ClientSession | None, optional): An existing aiohttp ClientSession to use. If not provided, a new session will be created.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        api_key = api_key or os.environ.get("SMALLEST_API_KEY")
        if not api_key:
            raise ValueError("SMALLEST_API_KEY must be set")

        self._opts = _TTSOptions(
            model=model,
            language=language,
            encoding=encoding,
            sample_rate=sample_rate,
            voice=voice,
            api_key=api_key,
            transliterate=transliterate,
            add_wav_header=add_wav_header,
            similarity=similarity,
            speed=speed,
            consistency=consistency,
            enhancement=enhancement,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "ChunkedStream":
        return ChunkedStream(
            tts=self,
            input_text=text,  # Changed from 'text' to 'input_text'
            conn_options=conn_options,
        )


class ChunkedStream(tts.ChunkedStream):
    """Synthesize chunked text using the Waves API endpoint"""

    def __init__(
        self,
        tts: TTS,
        input_text: str,  # Changed from 'text' to 'input_text'
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = tts._opts
        self._session = tts._ensure_session()

    @utils.log_exceptions(logger=logger)
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:  # Fixed: Added output_emitter parameter
        """Run the smallest.ai TTS request and emit audio via the output emitter."""
        
        # Initialize the output emitter
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/wav" if self._opts.add_wav_header else "audio/pcm",
        )

        self._chunk_size = 250
        if self._opts.model == "lightning-large":
            self._chunk_size = 140
        text_chunks = _split_into_chunks(self._input_text, self._chunk_size)

        for chunk in text_chunks:
            data = _to_smallest_options(self._opts)
            data["text"] = chunk

            url = f"{API_BASE_URL}/{self._opts.model}/get_speech"
            headers = {
                "Authorization": f"Bearer {self._opts.api_key}",
                "Content-Type": "application/json",
            }

            try:
                async with self._session.post(
                    url, 
                    headers=headers, 
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=self._conn_options.timeout)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"smallest.ai API error: {resp.status} - {error_text}")
                        raise APIStatusError(
                            message=f"smallest.ai API Error: {error_text}", 
                            status_code=resp.status
                        )

                    # Read and emit the audio data
                    audio_data = await resp.read()
                    output_emitter.push(audio_data)
                    
            except Exception as e:
                logger.error(f"Error in smallest.ai TTS: {e}")
                raise


def _to_smallest_options(opts: _TTSOptions) -> dict[str, Any]:
    return {
        "voice_id": opts.voice,
        "sample_rate": opts.sample_rate,
        "add_wav_header": opts.add_wav_header,
        "similarity": opts.similarity,
        "speed": opts.speed,
        "consistency": opts.consistency,
        "enhancement": opts.enhancement,
    }


def _split_into_chunks(text: str, chunk_size: int = 250) -> List[str]:
    chunks = []
    while text:
        if len(text) <= chunk_size:
            chunks.append(text.strip())
            break

        chunk_text = text[:chunk_size]
        last_break_index = -1

        # Find last sentence boundary using regex
        for i in range(len(chunk_text) - 1, -1, -1):
            if SENTENCE_END_REGEX.match(chunk_text[:i + 1]):
                last_break_index = i
                break

        if last_break_index == -1:
            # Fallback to space if no sentence boundary found
            last_space = chunk_text.rfind(' ')
            if last_space != -1:
                last_break_index = last_space 
            else:
                last_break_index = chunk_size - 1

        chunks.append(text[:last_break_index + 1].strip())
        text = text[last_break_index + 1:].strip()

    return chunks