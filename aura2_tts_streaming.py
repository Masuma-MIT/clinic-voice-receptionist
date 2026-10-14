import asyncio
import os

from dotenv import load_dotenv
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.speak.v1.types import (
    SpeakV1Text,
    SpeakV1Flushed,
)


load_dotenv()


TTS_MODEL = "aura-2-asteria-en"
TTS_SAMPLE_RATE = 24000
TTS_FIRST_AUDIO_TIMEOUT_SECONDS = 7
TTS_STREAM_GAP_TIMEOUT_SECONDS = 12


async def stream_aura2(
    text: str
):

    api_key = os.getenv(
        "DEEPGRAM_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "DEEPGRAM_API_KEY not found"
        )


    if not text or not text.strip():

        return


    deepgram = AsyncDeepgramClient(
        api_key=api_key
    )


    audio_queue = asyncio.Queue()

    listener_task = None

    error_holder = {
        "error": None
    }


    async with deepgram.speak.v1.connect(

        model=TTS_MODEL,
        encoding="linear16",
        sample_rate=TTS_SAMPLE_RATE

    ) as connection:


        def on_message(
            message
        ):

            if isinstance(
                message,
                bytes
            ):

                audio_queue.put_nowait(
                    (
                        "audio",
                        message
                    )
                )

                return


            if isinstance(
                message,
                SpeakV1Flushed
            ):

                audio_queue.put_nowait(
                    (
                        "done",
                        None
                    )
                )


        def on_error(
            error
        ):

            error_holder[
                "error"
            ] = error

            audio_queue.put_nowait(
                (
                    "error",
                    error
                )
            )


        connection.on(
            EventType.MESSAGE,
            on_message
        )


        connection.on(
            EventType.ERROR,
            on_error
        )


        listener_task = asyncio.create_task(
            connection.start_listening()
        )


        try:

            await asyncio.sleep(
                0.1
            )


            await connection.send_text(
                SpeakV1Text(
                    text=text
                )
            )


            await connection.send_flush()


            received_audio = False

            while True:

                item_type, payload = (
                    await asyncio.wait_for(
                        audio_queue.get(),
                        timeout=(
                            TTS_STREAM_GAP_TIMEOUT_SECONDS
                            if received_audio
                            else TTS_FIRST_AUDIO_TIMEOUT_SECONDS
                        )
                    )
                )


                if item_type == "audio":

                    received_audio = True

                    yield payload

                    continue


                if item_type == "done":

                    break


                if item_type == "error":

                    raise RuntimeError(
                        f"Aura-2 streaming error: "
                        f"{payload}"
                    )


        finally:

            try:

                await connection.send_close()

            except Exception:

                pass


            if listener_task is not None:

                listener_task.cancel()

                try:

                    await listener_task

                except asyncio.CancelledError:

                    pass

                except Exception:

                    pass
