from evdev import InputDevice, list_devices, ecodes
import select
import numpy as np
import mmap
import os
import struct

# =========================
# Encoder mapping
# =========================
ENCODER_MAPPING = {
    'rotary@18': 1,   # Motor 1 (rechts voor)
    'rotary@a': 0,   # Motor 0 (links voor)
    'rotary@8': 3,   # Motor 3 (rechts achter)
    'rotary@5': 2,  # Motor 2 (links achter)
}

num_encoders = len(ENCODER_MAPPING)

# =========================
# Detecteer devices
# =========================
devices = [InputDevice(path) for path in list_devices()]
rotary_devices = {dev.name: dev for dev in devices if 'rotary@' in dev.name}

print("Gevonden encoders:")
for i, dev in enumerate(rotary_devices):
    print(f"{i}: {dev}")

# Check of alle gewenste encoders gevonden zijn
for name in ENCODER_MAPPING.keys():
    if name not in rotary_devices:
        print(f"Waarschuwing: {name} niet gevonden in /dev/input!")

# Grab alle gevonden devices
for dev in rotary_devices.values():
    dev.grab()

# =========================
# POSIX shared memory met mmap
# =========================
shm_file_path = "/dev/shm/encoder_positions"

# maak bestand aan of open bestaand
fd = os.open(shm_file_path, os.O_CREAT | os.O_RDWR)

# zet de juiste grootte
shm_size = num_encoders * 8  # int64
os.ftruncate(fd, shm_size)

# memory-map het bestand
shared_mem = mmap.mmap(fd, shm_size, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ)

# sluit fd want mmap houdt referentie
os.close(fd)

# Helper functies
def read_position(index):
    shared_mem.seek(index * 8)
    return struct.unpack('q', shared_mem.read(8))[0]

def write_position(index, value):
    shared_mem.seek(index * 8)
    shared_mem.write(struct.pack('q', value))

# init posities
for i in range(num_encoders):
    write_position(i, 0)

print(f"POSIX shared memory aangemaakt in '{shm_file_path}' voor {num_encoders} encoders.")

# =========================
# Main loop: lees events en update positions
# =========================
try:
    while True:
        r, w, x = select.select(list(rotary_devices.values()), [], [])
        for dev in r:
            for event in dev.read():
                if event.type == ecodes.EV_REL:
                    index = ENCODER_MAPPING[dev.name]
                    current = read_position(index)
                    write_position(index, current + event.value)
                    # Debug print optioneel:
                    #print(f"{dev.name} -> index {index}, delta {event.value}, positie {current + event.value}")

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    for dev in rotary_devices.values():
        dev.ungrab()
    shared_mem.close()
    # bestand blijft bestaan in /dev/shm, kan door andere processen gelezen worden