import json
import queue
import subprocess
import threading
import time

def start_worker_reader(process):
    output = queue.Queue()
    stream = process.stdout
    def read():
        try:
            for line in stream:
                output.put(("line", line))
        except Exception as error:
            output.put(("error", type(error).__name__))
        finally:
            output.put(("eof", None))
            try:
                stream.close()
            except Exception:
                pass
    threading.Thread(target=read, daemon=True).start()
    return output

def stop_worker(process):
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)
    if process.stdin is not None:
        try:
            process.stdin.close()
        except Exception:
            pass

def request_worker(process, output, request, timeout, prefix="V3_RESULT "):
    deadline = time.monotonic() + max(0.01, float(timeout))
    try:
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Speaker worker response deadline exceeded.")
            try:
                kind, value = output.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError("Speaker worker response deadline exceeded.") from None
            if kind == "error":
                raise RuntimeError("Speaker worker output reader failed: " + value)
            if kind == "eof":
                raise RuntimeError("Speaker worker stopped unexpectedly.")
            if value.startswith("V3_TIMING "):
                # Forward performance-only worker events to the gateway console.
                # They remain separate from the V3_RESULT response protocol.
                print(value.rstrip(), flush=True)
                continue
            if value.startswith(prefix):
                result = json.loads(value[len(prefix):])
                if not isinstance(result, dict):
                    raise RuntimeError("Speaker worker returned an invalid result.")
                return result
    except Exception:
        stop_worker(process)
        raise
