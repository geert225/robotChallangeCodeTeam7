import subprocess
import threading
import signal
import sys
import time
import os

# Scripts in volgorde
PROCESS_COMMANDS = [
    ("ENCODERS", ["python", "../motors/encoders.py"]),
    ("MECANUM", ["python", "../motors/mecanum.py"]),
    ("CAMERA", ["python", "../vision/camera.py"]),
]

# Shutdown script
SHUTDOWN_COMMAND = ["python", "../motors/stop.py"]

processes = []
shutting_down = False


def stream_output(process, name):
    """Lees stdout/stderr en print met prefix"""
    for line in iter(process.stdout.readline, b''):
        print(f"[{name}] {line.decode().rstrip()}")
    for line in iter(process.stderr.readline, b''):
        print(f"[{name} ERROR] {line.decode().rstrip()}")


def start_processes():
    global processes
    for name, cmd in PROCESS_COMMANDS:
        cmd[1] = os.path.abspath(cmd[1])
        print(f"Starting: {name} -> {cmd}")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, text=False)
        processes.append((name, p))

        # Start thread om output te lezen
        threading.Thread(target=stream_output, args=(p, name), daemon=True).start()
        time.sleep(1)  # optioneel: kleine delay


def stop_processes():
    global processes
    print("Stopping all processes...")
    for name, p in processes:
        if p.poll() is None:
            print(f"Terminating {name} ({p.pid})")
            p.terminate()

    for name, p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"Force killing {name} ({p.pid})")
            p.kill()

    processes.clear()


def run_shutdown():
    print("Running shutdown process...")
    SHUTDOWN_COMMAND[1] = os.path.abspath(SHUTDOWN_COMMAND[1])
    subprocess.run(SHUTDOWN_COMMAND)


def handle_exit(signum, frame):
    global shutting_down
    if shutting_down:
        return
    shutting_down = True
    print(f"Received signal {signum}, shutting down...")
    stop_processes()
    run_shutdown()
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    start_processes()

    try:
        while True:
            time.sleep(1)
            for name, p in processes:
                if p.poll() is not None:
                    print(f"{name} stopped unexpectedly ({p.pid})")
                    handle_exit(signal.SIGTERM, None)
    except KeyboardInterrupt:
        handle_exit(signal.SIGINT, None)


if __name__ == "__main__":
    main()