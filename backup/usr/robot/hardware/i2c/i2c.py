import time
import mmap
import struct
import os
import fcntl

from i2c.master import I2CMaster
from i2c.pca9685 import PCA9685
from i2c.mpu6050 import MPU6050
from i2c.addon_board import AddonBoard

# ================= ADDR CONFIG =================
PCA9685_ADDR = 0x00
MPU6050_ADDR = 0x00
ADDONBOARD_ADDR = 0x00


# ================= SHM CONFIG =================

PWM_FORMAT   = "<16H"
ULTRA_FORMAT = "<ddHH"
LED_FORMAT   = "<7B"
SERVO_FORMAT = "<2B"
GYRO_FORMAT  = "<dddd"

SHM_CONFIG = {
    "/dev/shm/pwm_setpoints": PWM_FORMAT,
    "/dev/shm/ultrasoon": ULTRA_FORMAT,
    "/dev/shm/led_ctrl": LED_FORMAT,
    "/dev/shm/servo": SERVO_FORMAT,
    "/dev/shm/gyro": GYRO_FORMAT,
}

# ================= SHM HELPERS =================

def create_or_open_shm(path, fmt):
    size = struct.calcsize(fmt)

    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(b"\x00" * size)

    fd = open(path, "r+b")
    return mmap.mmap(fd.fileno(), size)


def shm_read(shm, fmt):
    fcntl.flock(shm.fileno(), fcntl.LOCK_SH)
    shm.seek(0)
    data = struct.unpack(fmt, shm.read())
    fcntl.flock(shm.fileno(), fcntl.LOCK_UN)
    return data


def shm_write(shm, fmt, values):
    fcntl.flock(shm.fileno(), fcntl.LOCK_EX)
    shm.seek(0)
    shm.write(struct.pack(fmt, *values))
    fcntl.flock(shm.fileno(), fcntl.LOCK_UN)


# ================= INIT =================

i2c = I2CMaster(1)
pwm = PCA9685(i2c, address=PCA9685_ADDR)
mpu = MPU6050(i2c, address=MPU6050_ADDR)
addon = AddonBoard(i2c, address=ADDONBOARD_ADDR)

pwm.set_pwm_freq(800)

# SHM openen / aanmaken
shms = {path: create_or_open_shm(path, fmt) for path, fmt in SHM_CONFIG.items()}

shm_pwm   = shms["/dev/shm/pwm_setpoints"]
shm_ultra = shms["/dev/shm/ultrasoon"]
shm_led   = shms["/dev/shm/led_ctrl"]
shm_servo = shms["/dev/shm/servo"]
shm_gyro  = shms["/dev/shm/gyro"]

# ================= CACHE =================

prev_pwm   = None
prev_led   = None
prev_servo = None

first_run = True

last_ultra_time = 0
ultra_interval = 0.1

last_led_toggle = 0
led_state = False


# ================= MAIN LOOP =================

while True:
    now = time.time()

    # ------------------------
    # PWM (change detect)
    # ------------------------
    pwm_values = shm_read(shm_pwm, PWM_FORMAT)

    if first_run or pwm_values != prev_pwm:
        for ch, val in enumerate(pwm_values):
            pwm.set_pwm(ch, val)

        prev_pwm = pwm_values

    # ------------------------
    # Servo (change detect)
    # ------------------------
    servo_values = shm_read(shm_servo, SERVO_FORMAT)

    if first_run or servo_values != prev_servo:
        addon.set_servo(*servo_values)
        prev_servo = servo_values

    # ------------------------
    # LED (change detect + mode logic)
    # ------------------------
    led_values = shm_read(shm_led, LED_FORMAT)

    if first_run or led_values != prev_led:
        prev_led = led_values

        mode, r1, g1, b1, r2, g2, b2 = led_values

        if mode == 0:
            addon.set_led(0, 0, 0)

        elif mode == 1:
            addon.set_led(r1, g1, b1)

        elif mode == 2:
            if now - last_led_toggle > 0.5:
                led_state = not led_state
                last_led_toggle = now

            addon.set_led(r1, g1, b1 if led_state else 0)

        elif mode == 3:
            if now - last_led_toggle > 0.5:
                led_state = not led_state
                last_led_toggle = now

            if led_state:
                addon.set_led(r1, g1, b1)
            else:
                addon.set_led(r2, g2, b2)

    # ------------------------
    # Ultrasonic
    # ------------------------
    ultra_values = shm_read(shm_ultra, ULTRA_FORMAT)
    update_rate = ultra_values[0]

    if update_rate > 0:
        ultra_interval = 1.0 / update_rate

    if now - last_ultra_time > ultra_interval:
        last_ultra_time = now

        try:
            d1, d2 = addon.read_ultrasonic()

            shm_write(
                shm_ultra,
                ULTRA_FORMAT,
                (update_rate, now, d1, d2)
            )

        except Exception as e:
            print("Ultrasonic error:", e)

    # ------------------------
    # Gyro (altijd updaten)
    # ------------------------
    try:
        gx, gy, gz = mpu.read_gyro_dps()

        shm_write(
            shm_gyro,
            GYRO_FORMAT,
            (now, gx, gy, gz)
        )

    except Exception as e:
        print("MPU error:", e)

    # ------------------------
    # First run done
    # ------------------------
    if first_run:
        first_run = False

    time.sleep(0.002)