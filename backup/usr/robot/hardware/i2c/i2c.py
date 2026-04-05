import time
import mmap
import struct
import os
import fcntl
import math

from i2c.master import I2CMaster
from i2c.pca9685 import PCA9685
from i2c.mpu6050 import MPU6050
from i2c.addonBoard import AddonBoard

# ================= CONFIG =================

PCA9685_ADDR = 0x43
MPU6050_ADDR = 0x68
ADDONBOARD_ADDR = 0x08

ENABLE_ADDON = True

# ================= SHM =================

PWM_FORMAT   = "<16H"
ULTRA_FORMAT = "<ddHH"
LED_FORMAT   = "<7B"
SERVO_FORMAT = "<B"
GYRO_FORMAT  = "<dddd"   # (timestamp, roll, pitch, yaw)   yaw in graden
ACCEL_FORMAT = "<ddd"    # (timestamp, ax_world [m/s²], ay_world [m/s²])
                         # wereldframe, zwaartekracht al afgetrokken

# ── As-mapping: pas aan als de sensor anders gemonteerd is ──────────────────
# +1 = as klopt, -1 = as omgekeerd.
# Rijrichting voor = +X in wereldframe, links = +Y in wereldframe.
ACCEL_SIGN_X =  1   # sensor-X richting t.o.v. robot-X (vooruit)
ACCEL_SIGN_Y =  1   # sensor-Y richting t.o.v. robot-Y (links)

SHM_CONFIG = {
    "/dev/shm/pwm_setpoints": PWM_FORMAT,
    "/dev/shm/ultrasoon":     ULTRA_FORMAT,
    "/dev/shm/led_ctrl":      LED_FORMAT,
    "/dev/shm/servo":         SERVO_FORMAT,
    "/dev/shm/gyro":          GYRO_FORMAT,
    "/dev/shm/mpu_accel":     ACCEL_FORMAT,
}

# ================= SHM HELPERS =================

def create_or_open_shm(path, fmt):
    size = struct.calcsize(fmt)

    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(b"\x00" * size)

    f = open(path, "r+b")
    shm = mmap.mmap(f.fileno(), size)
    return shm, f


def shm_read(shm, fd, fmt):
    fcntl.flock(fd.fileno(), fcntl.LOCK_SH)
    shm.seek(0)
    data = struct.unpack(fmt, shm.read())
    fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
    return data


def shm_write(shm, fd, fmt, values):
    fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
    shm.seek(0)
    shm.write(struct.pack(fmt, *values))
    fcntl.flock(fd.fileno(), fcntl.LOCK_UN)

# ================= MADGWICK FILTER =================

class Madgwick:
    def __init__(self, beta=0.1):
        self.beta = beta
        self.q0 = 1.0
        self.q1 = 0.0
        self.q2 = 0.0
        self.q3 = 0.0

    def update(self, gx, gy, gz, ax, ay, az, dt):
        q0, q1, q2, q3 = self.q0, self.q1, self.q2, self.q3

        # normalize accel
        norm = math.sqrt(ax*ax + ay*ay + az*az)
        if norm == 0:
            return
        ax /= norm
        ay /= norm
        az /= norm

        # gradient descent
        f1 = 2*(q1*q3 - q0*q2) - ax
        f2 = 2*(q0*q1 + q2*q3) - ay
        f3 = 2*(0.5 - q1*q1 - q2*q2) - az

        s0 = -2*q2*f1 + 2*q1*f2
        s1 = 2*q3*f1 + 2*q0*f2 - 4*q1*f3
        s2 = -2*q0*f1 + 2*q3*f2 - 4*q2*f3
        s3 = 2*q1*f1 + 2*q2*f2

        norm_s = math.sqrt(s0*s0 + s1*s1 + s2*s2 + s3*s3)
        if norm_s != 0:
            s0 /= norm_s
            s1 /= norm_s
            s2 /= norm_s
            s3 /= norm_s

        # integrate gyro
        gx = math.radians(gx)
        gy = math.radians(gy)
        gz = math.radians(gz)

        qDot0 = 0.5 * (-q1*gx - q2*gy - q3*gz) - self.beta * s0
        qDot1 = 0.5 * ( q0*gx + q2*gz - q3*gy) - self.beta * s1
        qDot2 = 0.5 * ( q0*gy - q1*gz + q3*gx) - self.beta * s2
        qDot3 = 0.5 * ( q0*gz + q1*gy - q2*gx) - self.beta * s3

        q0 += qDot0 * dt
        q1 += qDot1 * dt
        q2 += qDot2 * dt
        q3 += qDot3 * dt

        norm_q = math.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
        self.q0, self.q1, self.q2, self.q3 = (
            q0/norm_q, q1/norm_q, q2/norm_q, q3/norm_q
        )

    def get_euler(self):
        q0, q1, q2, q3 = self.q0, self.q1, self.q2, self.q3

        roll = math.degrees(math.atan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2)))
        pitch = math.degrees(math.asin(2*(q0*q2 - q3*q1)))
        yaw = math.degrees(math.atan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3)))

        return roll, pitch, yaw

# ================= INIT =================

i2c = I2CMaster(1)
pwm = PCA9685(i2c, address=PCA9685_ADDR)
mpu = MPU6050(i2c, address=MPU6050_ADDR)

if ENABLE_ADDON:
    addon = AddonBoard(i2c, address=ADDONBOARD_ADDR)

pwm.set_pwm_freq(800)

shms = {path: create_or_open_shm(path, fmt) for path, fmt in SHM_CONFIG.items()}

shm_pwm,   fd_pwm   = shms["/dev/shm/pwm_setpoints"]
shm_ultra, fd_ultra = shms["/dev/shm/ultrasoon"]
shm_led,   fd_led   = shms["/dev/shm/led_ctrl"]
shm_servo, fd_servo = shms["/dev/shm/servo"]
shm_gyro,  fd_gyro  = shms["/dev/shm/gyro"]
shm_accel, fd_accel = shms["/dev/shm/mpu_accel"]

# ================= GYRO CALIBRATION =================

print("Calibrating gyro... keep still")
samples = 500
bx = by = bz = 0

for _ in range(samples):
    gx, gy, gz = mpu.read_gyro_dps()
    bx += gx
    by += gy
    bz += gz
    time.sleep(0.002)

bias_gx = bx / samples
bias_gy = by / samples
bias_gz = bz / samples

print("Bias:", bias_gx, bias_gy, bias_gz)

# ================= FILTER =================

filter = Madgwick(beta=0.1)
last_time = time.time()

# ================= CACHE =================

prev_pwm   = None
prev_led   = None
prev_servo = None

first_run = True

last_ultra_time = 0
ultra_interval = 0.1

last_led_toggle = 0
led_state = False

# ================= LOOP =================

while True:
    now = time.time()
    dt = now - last_time

    if dt > 0.1:
        first_run = True
        last_time = now

    # ------------------------
    # PWM (change detect)
    # ------------------------
    pwm_values = shm_read(shm_pwm, fd_pwm, PWM_FORMAT)

    if first_run or pwm_values != prev_pwm:
        #for ch, val in enumerate(pwm_values):
        #    pwm.set_pwm(ch, val)
        pwm.set_all_pwm(pwm_values)
        prev_pwm = pwm_values


    if ENABLE_ADDON:
        # ------------------------
        # Servo (change detect)
        # ------------------------
        servo_values = shm_read(shm_servo, fd_servo, SERVO_FORMAT)

        if first_run or servo_values != prev_servo:
            #print(f"servo waarde {servo_values[0]}")
            addon.set_servo(servo_values[0])
            prev_servo = servo_values

        # ------------------------
        # LED (change detect + mode logic)
        # ------------------------
        led_values = shm_read(shm_led, fd_led, LED_FORMAT)

        if first_run or led_values != prev_led:
            prev_led = led_values

            mode, r1, g1, b1, r2, g2, b2 = led_values

            addon.set_led(mode, r1, g1, b1, r2, g2, b2)

        # ------------------------
        # Ultrasonic
        # ------------------------
        ultra_values = shm_read(shm_ultra, fd_ultra, ULTRA_FORMAT)
        update_rate = ultra_values[0]

        if update_rate > 0:
            ultra_interval = 1.0 / update_rate

        if now - last_ultra_time > ultra_interval:
            last_ultra_time = now

            try:
                d1, d2 = addon.read_ultrasonic()

                shm_write(
                    shm_ultra,
                    fd_ultra,
                    ULTRA_FORMAT,
                    (update_rate, now, d1, d2)
                )

            except Exception as e:
                print("Ultrasonic error:", e)

    # ------------------------
    # Gyro + Accel (altijd updaten)
    # ------------------------
    try:
        gx, gy, gz = mpu.read_gyro_dps()
        ax, ay, az = mpu.read_accel_g()

        # bias correctie
        gx -= bias_gx
        gy -= bias_gy
        gz -= bias_gz

        # Madgwick update (gebruikt genormeerde ax,ay,az intern)
        filter.update(gx, gy, gz, ax, ay, az, dt)

        roll, pitch, yaw = filter.get_euler()

        shm_write(shm_gyro, fd_gyro, GYRO_FORMAT, (now, roll, pitch, yaw))

        # ── Lineaire versnelling in wereldframe ──────────────────────────
        # Zwaartekrachtsrichting in sensorframe afleiden uit Madgwick-quaternion.
        # Gebaseerd op de gradiënt-vergelijkingen van Madgwick (f1,f2,f3):
        #   f1 = 2(q1q3 - q0q2) - ax_n  →  zwaartekracht-x in sensorframe
        #   f2 = 2(q0q1 + q2q3) - ay_n  →  zwaartekracht-y
        #   f3 = 2(0.5 - q1²  - q2²) - az_n  →  zwaartekracht-z
        q0, q1, q2, q3 = filter.q0, filter.q1, filter.q2, filter.q3

        gx_s = 2.0 * (q1*q3 - q0*q2)               # g-eenheden, sensorframe
        gy_s = 2.0 * (q0*q1 + q2*q3)
        gz_s = 1.0 - 2.0*q1*q1 - 2.0*q2*q2

        # Lineaire versnelling sensorframe [m/s²]: meet min zwaartekracht
        ax_l = (ax - gx_s) * 9.81
        ay_l = (ay - gy_s) * 9.81
        az_l = (az - gz_s) * 9.81

        # Roteer naar wereldframe met rotatiematrix uit quaternion
        # Alleen x en y zijn nodig voor een rijdende robot op een vlakke vloer
        ax_w = ((1 - 2*(q2*q2 + q3*q3)) * ax_l
                +      2*(q1*q2 - q0*q3)  * ay_l
                +      2*(q1*q3 + q0*q2)  * az_l) * ACCEL_SIGN_X

        ay_w = (       2*(q1*q2 + q0*q3)  * ax_l
                + (1 - 2*(q1*q1 + q3*q3)) * ay_l
                +      2*(q2*q3 - q0*q1)  * az_l) * ACCEL_SIGN_Y

        shm_write(shm_accel, fd_accel, ACCEL_FORMAT, (now, ax_w, ay_w))

    except Exception as e:
        print("MPU error:", e)

    time.sleep(0.002)
    first_run = False