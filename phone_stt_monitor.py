import asyncio

class SpeechListenerStopped(RuntimeError):
    pass

def check_listener(listener):
    if not listener.done():
        return
    if listener.cancelled():
        raise SpeechListenerStopped("Speech listener was cancelled.")
    error = listener.exception()
    if error is not None:
        raise SpeechListenerStopped("Speech listener failed.") from error
    raise SpeechListenerStopped("Speech listener closed.")

async def receive_phone_message(websocket, listener):
    check_listener(listener)
    receive_task = asyncio.create_task(websocket.receive_text())
    try:
        done, _ = await asyncio.wait(
            {receive_task, listener},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if receive_task in done:
            return await receive_task
        check_listener(listener)
    finally:
        if not receive_task.done():
            receive_task.cancel()
        try:
            await receive_task
        except (asyncio.CancelledError, Exception):
            pass
