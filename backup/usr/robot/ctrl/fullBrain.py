
# fullBrain.py
#
# Mini-states (robot_mode):
#   MANUAL  - bestuur via webinterface + gripper jog/trigger
#   AUTO    - autonome visie-logica (upgradedBasicBrain), webinterface blijft actief
#
# WebSocket commando's:
#   Rijden    : up / down / left / right / rot_left / rot_right / stop
#   Modus     : set_mode:manual  /  set_mode:auto
#   Gripper   : gripper_jog:<speed>  /  gripper_stop  /  gripper_auto
#
# Shared Memory:
#   /dev/shm/robot_cmd      <- mecanum rijcommando's  (4x double)
#   /dev/shm/gripper_cmd    <- gripper mode+speed     (<id>)
#   /dev/shm/vision_frame   <- camera frame           (timestamp + RGB)

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
# ROBOT MODUS
# ============================================================
MODE_MANUAL = "manual"
MODE_AUTO   = "auto"
robot_mode  = MODE_MANUAL          # beginstand = MANUAL

# ============================================================
# SHARED MEMORY — RIJCOMMANDO'S
# ============================================================
CMD_SHM_PATH = "/dev/shm/robot_cmd"
CMD_SIZE     = 32                  # 4x double

fd_cmd = os.open(CMD_SHM_PATH, os.O_CREAT | os.O_RDWR)
os.ftruncate(fd_cmd, CMD_SIZE)
cmd_mem = mmap.mmap(fd_cmd, CMD_SIZE, mmap.MAP_SHARED, mmap.PROT_WRITE)
os.close(fd_cmd)

def write_drive_cmd(vx, vy, omega):
    cmd_mem[:CMD_SIZE] = struct.pack("dddd", vx, vy, omega, time.time())

# ============================================================
# SHARED MEMORY — GRIPPER
# ============================================================
GRIPPER_IDLE = 0
GRIPPER_AUTO = 1
GRIPPER_JOG  = 2

GRIPPER_SHM_PATH = "/dev/shm/gripper_cmd"
GRIPPER_FMT      = "<id"
GRIPPER_SIZE     = struct.calcsize(GRIPPER_FMT)

def _create_or_open_shm(path, size):
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(b"\x00" * size)
    fh = open(path, "r+b")
    shm = mmap.mmap(fh.fileno(), size)
    return shm, fh

gripper_shm, gripper_fh = _create_or_open_shm(GRIPPER_SHM_PATH, GRIPPER_SIZE)

def write_gripper_cmd(mode, speed=0.0):
    fcntl.flock(gripper_fh.fileno(), fcntl.LOCK_EX)
    gripper_shm.seek(0)
    gripper_shm.write(struct.pack(GRIPPER_FMT, mode, speed))
    fcntl.flock(gripper_fh.fileno(), fcntl.LOCK_UN)

# ============================================================
# SHARED MEMORY — VISION FRAME
# ============================================================
WIDTH, HEIGHT, CHANNELS = 320, 240, 3
FRAME_SIZE          = WIDTH * HEIGHT * CHANNELS
VISION_FRAME_TOTAL  = 8 + FRAME_SIZE
VISION_FRAME_PATH   = "/dev/shm/vision_frame"

vision_shm, vision_fh = _create_or_open_shm(VISION_FRAME_PATH, VISION_FRAME_TOTAL)

latest_jpeg = None

# ============================================================
# MANUAL RIJSTATUS
# ============================================================
drive_state = {"vx": 0.0, "vy": 0.0, "omega": 0.0}
SPEED     = 1.5
ROT_SPEED = 3.0

# ============================================================
# AUTO — VISIE PARAMETERS  (uit upgradedBasicBrain)
# ============================================================
LOWER_HSV       = np.array([130, 140,  90])
UPPER_HSV       = np.array([150, 255, 220])
MIN_BEKER_AREA  = 50
MAX_BEKER_AREA  = 300_000
MIN_OBST_AREA   = 250

K_OMEGA         = 2.8
MAX_OMEGA       = 4.0
STOP_AREA       = 9_000
SEARCH_OMEGA    = 4.0
ERROR_DEADBAND  = 0.02
DRIVE_SPEED     = 2.0
DRIVE_MIN_OMEGA = 1.5
MAX_TRACK_DIST  = 80

_last_dir_pos = False
_last_target  = None          # (cx, cy) van vorige frame


def _classify_zone(cx, frame_w):
    if cx < frame_w * 0.33:
        return "LINKS"
    elif cx < frame_w * 0.66:
        return "MIDDEN"
    return "RECHTS"


def _compute_velocity(bekers, frame_w):
    global _last_dir_pos, _last_target

    if not bekers:
        _last_target = None
        return (0.0, 0.0, -SEARCH_OMEGA) if _last_dir_pos else (0.0, 0.0, SEARCH_OMEGA)

    # target tracking
    if _last_target is not None:
        hits = []
        for b in bekers:
            x, y, w, h, area = b
            cx, cy = x + w // 2, y + h // 2
            d2 = (cx - _last_target[0])**2 + (cy - _last_target[1])**2
            if d2 < MAX_TRACK_DIST**2:
                hits.append((d2, b))
        if hits:
            _, best = min(hits, key=lambda t: t[0])
        else:
            best = max(bekers, key=lambda b: b[4])
    else:
        best = max(bekers, key=lambda b: b[4])

    x, y, w, h, area = best
    cx = x + w // 2
    _last_target  = (cx, y + h // 2)
    _last_dir_pos = cx >= frame_w // 2

    error = (cx - frame_w // 2) / (frame_w // 2)
    omega = 0.0 if abs(error) < ERROR_DEADBAND else max(-MAX_OMEGA, min(MAX_OMEGA, -K_OMEGA * error))
    vx    = DRIVE_SPEED if abs(omega) < DRIVE_MIN_OMEGA else 0.0

    if area > STOP_AREA:
        return 0.0, 0.0, 0.0

    return vx, 0.0, omega

# ============================================================
# VISION LOOP  (altijd actief, schrijft rijcommando alleen in AUTO)
# ============================================================
async def vision_loop():
    global latest_jpeg

    while True:
        # frame lezen
        fcntl.flock(vision_fh.fileno(), fcntl.LOCK_SH)
        try:
            vision_shm.seek(0)
            vision_shm.read(8)                     # timestamp (overgeslagen)
            frame_bytes = vision_shm.read(FRAME_SIZE)
        finally:
            fcntl.flock(vision_fh.fileno(), fcntl.LOCK_UN)

        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, CHANNELS))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # bekerdetectie
        hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bekers = []
        for c in contours:
            area = cv2.contourArea(c)
            if MIN_BEKER_AREA < area < MAX_BEKER_AREA:
                x, y, w, h = cv2.boundingRect(c)
                bekers.append((x, y, w, h, area))
                cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # obstakeldetectie (onderste helft)
        roi   = frame_bgr[int(HEIGHT * 0.55):HEIGHT, :]
        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnt_obs, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        obstakels = {"LINKS": 0, "MIDDEN": 0, "RECHTS": 0}
        for c in cnt_obs:
            if cv2.contourArea(c) < MIN_OBST_AREA:
                continue
            ox, oy, ow, oh = cv2.boundingRect(c)
            if ow < 12 or oh < 6:
                continue
            zone = _classify_zone(ox + ow // 2, WIDTH)
            obstakels[zone] += 1
            cv2.rectangle(roi, (ox, oy), (ox+ow, oy+oh), (0, 0, 255), 2)
        frame_bgr[int(HEIGHT * 0.55):HEIGHT, :] = roi

        # rijcommando alleen in AUTO modus
        if robot_mode == MODE_AUTO:
            vx, vy, omega = _compute_velocity(bekers, WIDTH)
            write_drive_cmd(vx, vy, omega)
            label = f"AUTO  vx:{vx:.2f} vy:{vy:.2f} w:{omega:.2f} bekers:{len(bekers)}"
        else:
            label = f"MANUAL  bekers:{len(bekers)}"

        cv2.putText(frame_bgr, label, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # JPEG bouwen
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        latest_jpeg = buf.getvalue()

        await asyncio.sleep(0.02)

# ============================================================
# WEBSOCKET HELPERS
# ============================================================
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

def _ws_handshake(request):
    key = None
    for line in request.split("\r\n"):
        if "Sec-WebSocket-Key" in line:
            key = line.split(":", 1)[1].strip()
    accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
    return (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )

async def _ws_read_frame(reader):
    header = await reader.read(2)
    if not header:
        return None
    payload_len = header[1] & 127
    if payload_len == 126:
        ext = await reader.read(2)
        payload_len = int.from_bytes(ext, "big")
    elif payload_len == 127:
        ext = await reader.read(8)
        payload_len = int.from_bytes(ext, "big")
    mask    = await reader.read(4)
    payload = await reader.read(payload_len)
    return bytes(b ^ mask[i % 4] for i, b in enumerate(payload)).decode(errors="ignore")

async def _ws_send_binary(writer, data: bytes):
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

async def _ws_send_text(writer, text: str):
    data = text.encode()
    header = bytearray([0x81])
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

# ============================================================
# CONTROLE WEBSOCKET
# ============================================================
async def handle_control_ws(reader, writer, request):
    global robot_mode, drive_state

    writer.write(_ws_handshake(request).encode())
    await writer.drain()

    # stuur huidige modus direct na verbinding
    await _ws_send_text(writer, f"mode:{robot_mode}")

    try:
        while True:
            msg = await _ws_read_frame(reader)
            if msg is None:
                break

            # ---- MODUS WISSELEN ----
            if msg == "set_mode:manual":
                robot_mode = MODE_MANUAL
                drive_state = {"vx": 0.0, "vy": 0.0, "omega": 0.0}
                write_drive_cmd(0.0, 0.0, 0.0)
                print("[brain] → MANUAL")
                await _ws_send_text(writer, "mode:manual")

            elif msg == "set_mode:auto":
                robot_mode = MODE_AUTO
                drive_state = {"vx": 0.0, "vy": 0.0, "omega": 0.0}
                print("[brain] → AUTO")
                await _ws_send_text(writer, "mode:auto")

            # ---- RIJDEN (alleen in MANUAL) ----
            elif robot_mode == MODE_MANUAL:
                if msg == "up":
                    drive_state["vx"] = SPEED
                elif msg == "down":
                    drive_state["vx"] = -SPEED
                elif msg == "left":
                    drive_state["vy"] = SPEED
                elif msg == "right":
                    drive_state["vy"] = -SPEED
                elif msg == "rot_left":
                    drive_state["omega"] = ROT_SPEED
                elif msg == "rot_right":
                    drive_state["omega"] = -ROT_SPEED
                elif msg == "stop":
                    drive_state = {"vx": 0.0, "vy": 0.0, "omega": 0.0}

                # ---- GRIPPER (alleen in MANUAL) ----
                elif msg.startswith("gripper_jog:"):
                    try:
                        spd = float(msg.split(":", 1)[1])
                        write_gripper_cmd(GRIPPER_JOG, spd)
                    except ValueError:
                        pass

                elif msg == "gripper_stop":
                    write_gripper_cmd(GRIPPER_IDLE)

                elif msg == "gripper_auto":
                    write_gripper_cmd(GRIPPER_AUTO)
                    print("[brain] gripper AUTO gestart")

                write_drive_cmd(drive_state["vx"], drive_state["vy"], drive_state["omega"])

    except Exception as e:
        print(f"[control WS] fout: {e}")

    # verbinding verloren → stop robot veilig
    write_drive_cmd(0.0, 0.0, 0.0)
    write_gripper_cmd(GRIPPER_IDLE)
    writer.close()

# ============================================================
# CAMERA WEBSOCKET
# ============================================================
async def handle_camera_ws(reader, writer, request):
    writer.write(_ws_handshake(request).encode())
    await writer.drain()

    try:
        while True:
            if latest_jpeg is None:
                await asyncio.sleep(0.02)
                continue
            await _ws_send_binary(writer, latest_jpeg)
            await asyncio.sleep(0.05)
    except Exception:
        pass

    writer.close()

# ============================================================
# HTTP HANDLER
# ============================================================
async def handle_client(reader, writer):
    data    = await reader.read(2048)
    request = data.decode(errors="ignore")

    if "Upgrade: websocket" in request:
        if "/cam" in request:
            await handle_camera_ws(reader, writer, request)
        else:
            await handle_control_ws(reader, writer, request)
        return

    writer.write(HTML_PAGE)
    await writer.drain()
    writer.close()

# ============================================================
# HEARTBEAT  (refresht timestamp in MANUAL zodat mecanum niet time-out)
# ============================================================
async def heartbeat():
    while True:
        if robot_mode == MODE_MANUAL:
            write_drive_cmd(drive_state["vx"], drive_state["vy"], drive_state["omega"])
        await asyncio.sleep(0.05)

# ============================================================
# HTML PAGINA
# ============================================================
HTML_PAGE = b"""\
HTTP/1.1 200 OK
Content-Type: text/html; charset=utf-8

<!DOCTYPE html>
<html lang="nl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robot Control</title>
<style>
  body { font-family: sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 12px; }
  h2   { margin: 0 0 10px; }

  #modebadge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-weight: bold; font-size: 1.1em; margin-right: 10px;
  }
  .badge-manual { background: #0f3460; color: #e94560; }
  .badge-auto   { background: #0f5c2e; color: #4ade80; }

  button {
    padding: 8px 16px; margin: 4px; border: none; border-radius: 6px;
    cursor: pointer; font-size: 0.95em;
  }
  .btn-switch  { background: #e94560; color: #fff; }
  .btn-switch.auto { background: #4ade80; color: #111; }
  .btn-drive   { background: #0f3460; color: #eee; font-size: 1.3em; width: 60px; height: 54px; }
  .btn-gripper { background: #2d2d54; color: #eee; }
  .btn-auto-trigger { background: #f59e0b; color: #111; font-weight: bold; }

  #controls { margin: 10px 0; }
  #drive-grid { display: grid; grid-template-columns: repeat(3, 60px); gap: 4px; margin: 8px 0; }
  .drive-center { grid-column: 2; }

  #gripper-panel { margin: 10px 0; }
  #status { font-size: 0.85em; color: #aaa; margin: 6px 0; }
  #cam    { display: block; margin-top: 10px; border: 2px solid #444; border-radius: 6px; }

  .hidden { display: none !important; }
  #auto-info { color: #4ade80; font-style: italic; margin: 6px 0; }
</style>
</head>
<body>

<h2>
  <span id="modebadge" class="badge-manual">MANUAL</span>
  Robot Control
</h2>

<div id="status">Verbinden...</div>

<!-- Modus wisselen -->
<div>
  <button id="btn-switch" class="btn-switch" onclick="switchMode()">Start AUTO</button>
</div>

<!-- AUTO info -->
<div id="auto-info" class="hidden">Robot rijdt autonoom op vision.</div>

<!-- MANUAL rijbediening -->
<div id="controls">
  <div><strong>Rijden</strong> <small>(pijltjes of knoppen)</small></div>
  <div id="drive-grid">
    <div></div>
    <button class="btn-drive" onmousedown="press('up')"    onmouseup="release()" ontouchstart="press('up')"    ontouchend="release()">&#8593;</button>
    <div></div>
    <button class="btn-drive" onmousedown="press('left')"  onmouseup="release()" ontouchstart="press('left')"  ontouchend="release()">&#8592;</button>
    <button class="btn-drive" onmousedown="press('stop')"  onmouseup="release()" ontouchstart="press('stop')"  ontouchend="release()">&#9632;</button>
    <button class="btn-drive" onmousedown="press('right')" onmouseup="release()" ontouchstart="press('right')" ontouchend="release()">&#8594;</button>
    <button class="btn-drive" onmousedown="press('rot_left')"  onmouseup="release()" ontouchstart="press('rot_left')"  ontouchend="release()">&#8634;</button>
    <button class="btn-drive" onmousedown="press('down')"  onmouseup="release()" ontouchstart="press('down')"  ontouchend="release()">&#8595;</button>
    <button class="btn-drive" onmousedown="press('rot_right')" onmouseup="release()" ontouchstart="press('rot_right')" ontouchend="release()">&#8635;</button>
  </div>
</div>

<!-- MANUAL gripper bediening -->
<div id="gripper-panel">
  <div><strong>Gripper</strong></div>
  <button class="btn-gripper"
    onmousedown="send('gripper_jog:60')"  onmouseup="send('gripper_stop')"
    ontouchstart="send('gripper_jog:60')" ontouchend="send('gripper_stop')">
    &#9650; Open (jog)
  </button>
  <button class="btn-gripper"
    onmousedown="send('gripper_jog:-60')" onmouseup="send('gripper_stop')"
    ontouchstart="send('gripper_jog:-60')" ontouchend="send('gripper_stop')">
    &#9660; Dicht (jog)
  </button>
  <button class="btn-auto-trigger" onclick="send('gripper_auto')">
    &#9654; Gripper AUTO
  </button>
</div>

<!-- Camera -->
<img id="cam" width="320" alt="camera">

<script>
let ws     = new WebSocket("ws://" + location.host + "/ws");
let camWs  = new WebSocket("ws://" + location.host + "/cam");
let mode   = "manual";
let keys   = new Set();
let pressTimer = null;

camWs.binaryType = "blob";
camWs.onmessage  = (e) => {
  let url = URL.createObjectURL(e.data);
  document.getElementById("cam").src = url;
};

ws.onopen    = () => setStatus("Verbonden");
ws.onclose   = () => setStatus("Verbinding verbroken");
ws.onerror   = () => setStatus("Verbindingsfout");
ws.onmessage = (e) => {
  if (e.data.startsWith("mode:")) {
    applyMode(e.data.split(":")[1]);
  }
};

function send(cmd) {
  if (ws.readyState === WebSocket.OPEN) ws.send(cmd);
}

function setStatus(msg) {
  document.getElementById("status").textContent = msg;
}

// Houd rijknop ingedrukt -> blijf commando sturen
function press(cmd) {
  if (mode !== "manual") return;
  send(cmd);
  pressTimer = setInterval(() => send(cmd), 80);
}
function release() {
  clearInterval(pressTimer);
  send("stop");
}

function switchMode() {
  if (mode === "manual") {
    send("set_mode:auto");
  } else {
    send("set_mode:manual");
  }
}

function applyMode(newMode) {
  mode = newMode;
  let badge   = document.getElementById("modebadge");
  let btn     = document.getElementById("btn-switch");
  let ctrl    = document.getElementById("controls");
  let gripper = document.getElementById("gripper-panel");
  let autoInfo= document.getElementById("auto-info");

  if (mode === "auto") {
    badge.textContent = "AUTO";
    badge.className   = "badge-auto";
    btn.textContent   = "Stop AUTO";
    btn.className     = "btn-switch auto";
    ctrl.classList.add("hidden");
    gripper.classList.add("hidden");
    autoInfo.classList.remove("hidden");
  } else {
    badge.textContent = "MANUAL";
    badge.className   = "badge-manual";
    btn.textContent   = "Start AUTO";
    btn.className     = "btn-switch";
    ctrl.classList.remove("hidden");
    gripper.classList.remove("hidden");
    autoInfo.classList.add("hidden");
  }
}

// Toetsenbord rijbesturing (MANUAL)
document.addEventListener("keydown", (e) => {
  if (mode !== "manual" || e.repeat) return;
  const map = {
    ArrowUp: "up", ArrowDown: "down",
    ArrowLeft: "left", ArrowRight: "right",
    "[": "rot_left", "]": "rot_right"
  };
  if (map[e.key]) { keys.add(e.key); flushKeys(); }
});
document.addEventListener("keyup", (e) => {
  keys.delete(e.key);
  flushKeys();
});
function flushKeys() {
  send("stop");
  if (keys.has("ArrowUp"))    send("up");
  if (keys.has("ArrowDown"))  send("down");
  if (keys.has("ArrowLeft"))  send("left");
  if (keys.has("ArrowRight")) send("right");
  if (keys.has("["))          send("rot_left");
  if (keys.has("]"))          send("rot_right");
}
</script>
</body>
</html>
"""

# ============================================================
# MAIN
# ============================================================
async def main():
    server = await asyncio.start_server(handle_client, "0.0.0.0", 8765)
    print("[brain] Server actief op http://ROBOT_IP:8765")
    print("[brain] Start in MANUAL modus")

    async with server:
        await asyncio.gather(
            server.serve_forever(),
            vision_loop(),
            heartbeat(),
        )

if __name__ == "__main__":
    asyncio.run(main())
