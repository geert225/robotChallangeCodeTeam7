import os
import time
import mmap
import struct
import fcntl
import asyncio
import io
import base64
import hashlib
from PIL import Image
import numpy as np
import cv2

# ============================================================
# SHARED MEMORY CONFIG
# ============================================================

WIDTH, HEIGHT, CHANNELS = 320, 240, 3
FRAME_SIZE = WIDTH * HEIGHT * CHANNELS
VISION_FRAME_TOTAL = 8 + FRAME_SIZE  # timestamp + frame
VISION_FRAME_PATH = "/dev/shm/vision_frame"

CMD_SHM_PATH = "/dev/shm/robot_cmd"
CMD_SIZE = 32

# open SHM for frames (read-only)
fd_frame = os.open(VISION_FRAME_PATH, os.O_RDONLY)
vision_frame_mm = mmap.mmap(fd_frame, VISION_FRAME_TOTAL, mmap.MAP_SHARED, mmap.PROT_READ)

# open SHM for commands
fd_cmd = os.open(CMD_SHM_PATH, os.O_CREAT | os.O_RDWR)
os.ftruncate(fd_cmd, CMD_SIZE)
cmd_mem = mmap.mmap(fd_cmd, CMD_SIZE, mmap.MAP_SHARED, mmap.PROT_WRITE)

latest_jpeg = None

# ============================================================
# DETECTIE PARAMETERS
# ============================================================

LOWER_HSV = np.array([130, 140, 90])
UPPER_HSV = np.array([150, 255, 220])
MIN_BEKER_AREA = 50
MAX_BEKER_AREA = 300000
MIN_OBSTAKEL_AREA = 250

BASE_SPEED = 0.4
MIN_SPEED = 0.2
K_OMEGA = 1.2
MAX_OMEGA = 0.8
STOP_AREA = 9000
SEARCH_OMEGA = -0.6
ERROR_DEADBAND = 0.05

def classify_zone(x_center, frame_width):
    if x_center < frame_width * 0.33:
        return "LINKS"
    elif x_center < frame_width * 0.66:
        return "MIDDEN"
    else:
        return "RECHTS"

# ============================================================
# HELPERS
# ============================================================

def write_command(vx, vy, omega):
    cmd_mem[:CMD_SIZE] = struct.pack("dddd", vx, vy, omega, time.time())

def compute_velocity(bekers, frame_width):
    if not bekers:
        return 0.0, 0.0, SEARCH_OMEGA
    # grootste beker kiezen
    x, y, w, h, area = max(bekers, key=lambda b: b[4])
    cx = x + w // 2
    error = (cx - frame_width // 2) / (frame_width // 2)
    if abs(error) < ERROR_DEADBAND:
        error = 0
    omega = -K_OMEGA * error
    omega = max(-MAX_OMEGA, min(MAX_OMEGA, omega))
    speed_factor = max(0, 1 - 1.5 * abs(error))
    vx = BASE_SPEED * speed_factor
    if vx < MIN_SPEED:
        vx = MIN_SPEED
    if area > STOP_AREA:
        return 0.0, 0.0, 0.0
    return 0.0, 0.0, omega

# ============================================================
# VISION LOOP (LEES SHM FRAMES + DETECTIE + WEB JPEG)
# ============================================================

async def vision_loop():
    global latest_jpeg
    while True:
        fcntl.flock(fd_frame, fcntl.LOCK_SH)
        try:
            vision_frame_mm.seek(0)
            timestamp = struct.unpack("d", vision_frame_mm.read(8))[0]
            frame_bytes = vision_frame_mm.read(FRAME_SIZE)
        finally:
            fcntl.flock(fd_frame, fcntl.LOCK_UN)

        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, CHANNELS))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # -------- BEKERDETECTIE --------
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bekers = [(x, y, w, h, cv2.contourArea(c)) 
                  for c in contours 
                  if MIN_BEKER_AREA < cv2.contourArea(c) < MAX_BEKER_AREA
                  for x, y, w, h in [cv2.boundingRect(c)]]

        # annotaties op frame
        for x, y, w, h, area in bekers:
            cx = x + w // 2
            cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame_bgr, f"BEKER", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

        # -------- OBSTAKELDETECTIE (ROI onder) --------
        roi = frame_bgr[int(HEIGHT*0.55):HEIGHT, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 40, 120)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obstakels = {"LINKS":0, "MIDDEN":0, "RECHTS":0}
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_OBSTAKEL_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w<12 or h<6:
                continue
            cx = x + w//2
            zone = classify_zone(cx, WIDTH)
            obstakels[zone] += 1
            cv2.rectangle(roi, (x,y), (x+w,y+h), (0,0,255), 2)
        frame_bgr[int(HEIGHT*0.55):HEIGHT, :] = roi

        # -------- COMPUTE COMMAND --------
        vx, vy, omega = compute_velocity(bekers, WIDTH)
        write_command(vx, vy, omega)

        # -------- JPEG voor web --------
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        latest_jpeg = buf.getvalue()

        await asyncio.sleep(0.02)

# ============================================================
# WEBSERVER + WEBSOCKET
# ============================================================

HTML_PAGE = b"""\
HTTP/1.1 200 OK
Content-Type: text/html

<!DOCTYPE html>
<html>
<head><title>Robot Vision</title></head>
<body>
<h2>Robot Vision</h2>
<img id="cam" width="320">
<script>
let cam_ws = new WebSocket("ws://" + location.host + "/cam");
cam_ws.binaryType = "blob";
cam_ws.onmessage = (event) => {
    document.getElementById("cam").src = URL.createObjectURL(event.data);
};
</script>
</body>
</html>
"""

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

def websocket_handshake(request):
    key = None
    for line in request.split("\r\n"):
        if "Sec-WebSocket-Key" in line:
            key = line.split(":")[1].strip()
    accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
    return (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )

async def handle_client(reader, writer):
    data = await reader.read(1024)
    request = data.decode(errors="ignore")
    if "Upgrade: websocket" in request and "/cam" in request:
        await handle_camera_ws(reader, writer, request)
        return
    writer.write(HTML_PAGE)
    await writer.drain()
    writer.close()

async def handle_camera_ws(reader, writer, request):
    writer.write(websocket_handshake(request).encode())
    await writer.drain()
    try:
        while True:
            if latest_jpeg is None:
                await asyncio.sleep(0.05)
                continue
            data = latest_jpeg
            header = bytearray([0x82])
            length = len(data)
            if length < 126:
                header.append(length)
            elif length < 65536:
                header.append(126)
                header += length.to_bytes(2, "big")
            else:
                header.append(127)
                header += length.to_bytes(8, "big")
            writer.write(header + data)
            await writer.drain()
            await asyncio.sleep(0.05)
    except:
        pass
    writer.close()

# ============================================================
# MAIN
# ============================================================

async def main():
    server = await asyncio.start_server(handle_client, "0.0.0.0", 8765)
    print("Server running on http://ROBOT_IP:8765")
    async with server:
        await asyncio.gather(
            server.serve_forever(),
            vision_loop()
        )

if __name__ == "__main__":
    asyncio.run(main())