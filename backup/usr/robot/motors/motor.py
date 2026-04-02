import mmap
import struct
import os

PWM_FORMAT = "<16H"
PWM_PATH = "/dev/shm/pwm_setpoints"
PWM_SIZE = struct.calcsize(PWM_FORMAT)

motorMapping = [
    (3, 2),   # Motor 0
    (0, 1),   # Motor 1 (rechts voor)
    (7, 6),   # Motor 2 
    (4, 5),   # Motor 3 (rechts achter)
    (9, 8)    # Motor 4 (lai klep)
]

# Open shared memory
def open_pwm_shm():
    fd = os.open(PWM_PATH, os.O_RDWR)
    return mmap.mmap(fd, PWM_SIZE)

pwm_shm = open_pwm_shm()


def read_pwm():
    pwm_shm.seek(0)
    data = pwm_shm.read(PWM_SIZE)
    return list(struct.unpack(PWM_FORMAT, data))


def write_pwm(values):
    pwm_shm.seek(0)
    pwm_shm.write(struct.pack(PWM_FORMAT, *values))


def set_motor(motor, procent):
    if motor < 0 or motor > 3:
        raise ValueError("Motor index must be between 0 and 3")
    if procent < -100 or procent > 100:
        raise ValueError("Procent must be between -100 and 100")

    pwm_values = read_pwm()

    ch_a, ch_b = motorMapping[motor]

    if procent < 0:
        duty = int(-procent / 100 * 4095)
        pwm_values[ch_a] = 0
        pwm_values[ch_b] = duty
    else:
        duty = int(procent / 100 * 4095)
        pwm_values[ch_a] = duty
        pwm_values[ch_b] = 0

    write_pwm(pwm_values)

def set_rad(procent):
    if procent < -100 or procent > 100:
        raise ValueError("Procent must be between -100 and 100")

    pwm_values = read_pwm()

    ch_a, ch_b = motorMapping[4]

    if procent < 0:
        duty = int(-procent / 100 * 4095)
        pwm_values[ch_a] = 0
        pwm_values[ch_b] = duty
    else:
        duty = int(procent / 100 * 4095)
        pwm_values[ch_a] = duty
        pwm_values[ch_b] = 0

    write_pwm(pwm_values)

def rem_rad():

    pwm_values = read_pwm()

    ch_a, ch_b = motorMapping[4]

    pwm_values[ch_a] = 4096
    pwm_values[ch_b] = 4096 

    write_pwm(pwm_values)

def rem_motor(motor):
    if motor < 0 or motor > 3:
        raise ValueError("Motor index must be between 0 and 3")

    pwm_values = read_pwm()

    ch_a, ch_b = motorMapping[motor]

    pwm_values[ch_a] = 4095
    pwm_values[ch_b] = 4095

    write_pwm(pwm_values)