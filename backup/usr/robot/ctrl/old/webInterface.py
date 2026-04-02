import asyncio
import mmap
import os
import struct
import time

# =========================
# SHARED MEMORY
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
# STATE
# =========================
state = {"vx": 0.0, "vy": 0.0, "omega": 0.0}

SPEED = 1.5
ROT_SPEED = 3.0

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

<script>
let ws = new WebSocket("ws://" + location.host + "/ws");

// actieve toetsen bijhouden
let keys = new Set();

function sendState() {
    if (keys.size === 0) {
        ws.send("stop");
        return;
    }

    // reset eerst
    ws.send("stop");

    // beweging
    if (keys.has("ArrowUp")) ws.send("up");
    if (keys.has("ArrowDown")) ws.send("down");
    if (keys.has("ArrowLeft")) ws.send("left");
    if (keys.has("ArrowRight")) ws.send("right");

    // rotatie
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
</script>

</body>
</html>
"""

# =========================
# HTTP + WS HANDLER
# =========================
async def handle_client(reader, writer):
    data = await reader.read(1024)
    request = data.decode(errors="ignore")

    if "Upgrade: websocket" in request:
        await handle_websocket(reader, writer, request)
        return

    writer.write(HTML_PAGE)
    await writer.drain()
    writer.close()

# =========================
# WEBSOCKET
# =========================
import base64
import hashlib

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

async def handle_websocket(reader, writer, request):
    global state

    key = None
    for line in request.split("\r\n"):
        if "Sec-WebSocket-Key" in line:
            key = line.split(":")[1].strip()

    accept = base64.b64encode(
        hashlib.sha1((key + GUID).encode()).digest()
    ).decode()

    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )
    writer.write(response.encode())
    await writer.drain()

    print("WebSocket connected")

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

            # ===== COMMANDS =====
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

    print("WebSocket disconnected")
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
        await asyncio.gather(server.serve_forever(), heartbeat())

asyncio.run(main())