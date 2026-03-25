import asyncio
import mmap
import os
import struct
import time
import base64
import hashlib
import io
import fcntl

import numpy as np
from PIL import Image

# =========================
# SHARED MEMORY (COMMANDS)
# =========================
CMD_SHM_PATH = "/dev/shm/robot_cmd"
CMD_SIZE = 32

fd = os.open(CMD_SHM_PATH, os.O_CREAT | os.O_RDWR)
os.ftruncate(fd, CMD_SIZE)
cmd_mem = mmap.mmap(fd, CMD_SIZE, mmap.MAP_SHARED, mmap.PROT_WRITE)
os.close(fd)

def write_command(vx, vy, omega):
    timestamp = time.time()
    cmd_mem[:CMD_SIZE] = struct.pack("dddd", vx, vy, omega, timestamp)

# =========================
# SHARED MEMORY (VISION)
# =========================
VISION_PATH = "/dev/shm/vision_frame"

WIDTH = 320
HEIGHT = 240
CHANNELS = 3

FRAME_SIZE = WIDTH * HEIGHT * CHANNELS
TOTAL_SIZE = 8 + FRAME_SIZE

vision_fd = os.open(VISION_PATH, os.O_RDONLY)
vision_mm = mmap.mmap(vision_fd, TOTAL_SIZE, mmap.MAP_SHARED, mmap.PROT_READ)

# =========================
# STATE
# =========================
state = {"vx": 0.0, "vy": 0.0, "omega": 0.0}

SPEED = 1.5
ROT_SPEED = 3.0

latest_jpeg = None

# =========================
# CAMERA LOOP (FROM SHM)
# =========================
async def camera_loop():
    global latest_jpeg

    while True:
        fcntl.flock(vision_fd, fcntl.LOCK_SH)

        try:
            vision_mm.seek(0)

            timestamp = struct.unpack('d', vision_mm.read(8))[0]
            frame_bytes = vision_mm.read(FRAME_SIZE)

        finally:
            fcntl.flock(vision_fd, fcntl.LOCK_UN)

        frame = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = frame.reshape((HEIGHT, WIDTH, CHANNELS))

        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=70)

        latest_jpeg = buf.getvalue()

        await asyncio.sleep(0.4)

# =========================
# HTML PAGE
# =========================
HTML_PAGE = b"""\
HTTP/1.1 200 OK
Content-Type: text/html

<!DOCTYPE html>
<html>
<head>
<title>Robot Control</title>
</head>
<body>
<h2>Robot Control</h2>

<p>Arrow keys = bewegen</p>
<p>[ ] = rotatie</p>

<img id="cam" width="320">

<script>
let ws = new WebSocket("ws://" + location.host + "/ws");
let cam_ws = new WebSocket("ws://" + location.host + "/cam");

cam_ws.binaryType = "blob";

let keys = new Set();

function sendState() {
    if (keys.size === 0) {
        ws.send("stop");
        return;
    }

    ws.send("stop");

    if (keys.has("ArrowUp")) ws.send("up");
    if (keys.has("ArrowDown")) ws.send("down");
    if (keys.has("ArrowLeft")) ws.send("left");
    if (keys.has("ArrowRight")) ws.send("right");

    if (keys.has("[")) ws.send("rot_left");
    if (keys.has("]")) ws.send("rot_right");
}

document.addEventListener("keydown", (e) => {
    if (e.repeat) return;

    if (["ArrowUp","ArrowDown","ArrowLeft","ArrowRight","[","]"].includes(e.key)) {
        keys.add(e.key);
        sendState();
    }
});

document.addEventListener("keyup", (e) => {
    keys.delete(e.key);
    sendState();
});

cam_ws.onmessage = (event) => {
    let url = URL.createObjectURL(event.data);
    document.getElementById("cam").src = url;
};
</script>

</body>
</html>
"""

# =========================
# WS HANDSHAKE
# =========================
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

def websocket_handshake(request):
    key = None
    for line in request.split("\r\n"):
        if "Sec-WebSocket-Key" in line:
            key = line.split(":")[1].strip()

    accept = base64.b64encode(
        hashlib.sha1((key + GUID).encode()).digest()
    ).decode()

    return (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )

# =========================
# CLIENT HANDLER
# =========================
async def handle_client(reader, writer):
    data = await reader.read(1024)
    request = data.decode(errors="ignore")

    if "Upgrade: websocket" in request:
        if "/cam" in request:
            await handle_camera_ws(reader, writer, request)
        else:
            await handle_websocket(reader, writer, request)
        return

    writer.write(HTML_PAGE)
    await writer.drain()
    writer.close()

# =========================
# CONTROL WS
# =========================
async def handle_websocket(reader, writer, request):
    global state

    writer.write(websocket_handshake(request).encode())
    await writer.drain()

    try:
        while True:
            data = await reader.read(2)
            if not data:
                break

            payload_len = data[1] & 127

            if payload_len == 126:
                data += await reader.read(2)
                payload_len = int.from_bytes(data[2:4], 'big')
                mask = await reader.read(4)
            elif payload_len == 127:
                data += await reader.read(8)
                payload_len = int.from_bytes(data[2:10], 'big')
                mask = await reader.read(4)
            else:
                mask = await reader.read(4)

            payload = await reader.read(payload_len)
            decoded = bytes(b ^ mask[i % 4] for i, b in enumerate(payload)).decode()

            if decoded == "up":
                state["vx"] = SPEED
            elif decoded == "down":
                state["vx"] = -SPEED
            elif decoded == "left":
                state["vy"] = SPEED
            elif decoded == "right":
                state["vy"] = -SPEED
            elif decoded == "rot_left":
                state["omega"] = ROT_SPEED
            elif decoded == "rot_right":
                state["omega"] = -ROT_SPEED
            elif decoded == "stop":
                state = {"vx": 0.0, "vy": 0.0, "omega": 0.0}

            write_command(state["vx"], state["vy"], state["omega"])

    except Exception as e:
        print("WS error:", e)

    writer.close()

# =========================
# CAMERA WS
# =========================
async def handle_camera_ws(reader, writer, request):
    writer.write(websocket_handshake(request).encode())
    await writer.drain()

    try:
        while True:
            if latest_jpeg is None:
                await asyncio.sleep(0.1)
                continue

            data = latest_jpeg

            header = bytearray()
            header.append(0x82)

            length = len(data)

            if length < 126:
                header.append(length)
            elif length < 65536:
                header.append(126)
                header += length.to_bytes(2, 'big')
            else:
                header.append(127)
                header += length.to_bytes(8, 'big')

            writer.write(header + data)
            await writer.drain()

            await asyncio.sleep(0.4)

    except Exception as e:
        print("Camera WS error:", e)

    writer.close()

# =========================
# HEARTBEAT
# =========================
async def heartbeat():
    while True:
        write_command(state["vx"], state["vy"], state["omega"])
        await asyncio.sleep(0.05)

# =========================
# MAIN
# =========================
async def main():
    server = await asyncio.start_server(handle_client, "0.0.0.0", 8765)

    print("Server running on http://ROBOT_IP:8765")

    async with server:
        await asyncio.gather(
            server.serve_forever(),
            heartbeat(),
            camera_loop()
        )

asyncio.run(main())