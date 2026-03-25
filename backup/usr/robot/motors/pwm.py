from smbus2 import SMBus
import time

# PCA9685 I2C address
PCA_ADDR = 0x60

# Registers
MODE1      = 0x00
PRESCALE   = 0xFE
LED0_ON_L  = 0x06

bus = SMBus(1)

def write(reg, val):
    bus.write_byte_data(PCA_ADDR, reg, val)
    # direct read back
    time.sleep(0.005)
    read_val = bus.read_byte_data(PCA_ADDR, reg)
    if read_val != val:
        print(f"Warning: reg 0x{reg:02X} wrote 0x{val:02X} but read 0x{read_val:02X}")
    else:
        print(f"OK: reg 0x{reg:02X} = 0x{read_val:02X}")
    return read_val    

# Set PWM frequency
def set_pwm_freq(freq_hz):
    #prescale_val = int(25000000.0 / (4096 * freq_hz) - 1)
    #oldmode = bus.read_byte_data(PCA_ADDR, MODE1)
    #write(MODE1, (oldmode & 0x7F) | 0x10)  # sleep
    #write(PRESCALE, prescale_val)
    #write(MODE1, oldmode)
    #time.sleep(0.005)
    #write(MODE1, oldmode | 0x80)

    osc_hz = 25_000_000

    # Calculate prescale with proper rounding (datasheet recommends round())
    prescale = round(osc_hz / (4096 * freq_hz)) - 1

    if not 3 <= prescale <= 255:
        raise ValueError(f"Invalid prescale value: {prescale}")

    actual_freq = osc_hz / (4096 * (prescale + 1))
    freq_error_pct = abs(actual_freq - freq_hz) / freq_hz * 100

    # 1. Put device in known state (reset MODE1)
    # AI=1 (auto increment), ALLCALL=1, everything else 0
    mode1_init = 0x21  # 0010 0001
    write(MODE1, mode1_init)

    # 2. Enter sleep (set SLEEP bit)
    write(MODE1, mode1_init | 0x10)

    # 3. Set prescale
    write(PRESCALE, prescale)

    # 4. Wake up (clear SLEEP)
    write(MODE1, mode1_init)
    time.sleep(0.005)

    # 5. Restart (set RESTART bit)
    write(MODE1, mode1_init | 0x80)

    # Debug readback
    mode1 = bus.read_byte_data(PCA_ADDR, MODE1)
    print("MODE1:", bin(mode1))
    print(f"Prescale: {prescale}  |  Requested: {freq_hz} Hz  |  Actual: {actual_freq:.1f} Hz  ({freq_error_pct:.1f}% off)")
    if freq_error_pct > 3:
        print(f"WARNING: PWM frequency error is {freq_error_pct:.1f}% — internal oscillator may need calibration")



# Set PWM for a channel
def set_pwm_percent(channel, percent):
    if percent < 0: percent = 0
    if percent > 100: percent = 100

    # converteer percentage naar 12-bit waarde
    off_count = int(4095 * (percent / 100.0))
    on_count = 0  # meestal start bij 0

    reg = LED0_ON_L + 4 * channel
    write(reg + 0, on_count & 0xFF)        # LED ON L
    write(reg + 1, on_count >> 8)      # LED ON H
    write(reg + 2, off_count & 0xFF)   # LED OFF L
    write(reg + 3, off_count >> 8)     # LED OFF H


def set_pwm(channel, duty):
    duty = max(0, min(4095, duty))
    reg = LED0_ON_L + 4 * channel
    #print(f"set pwmChannel: {channel}  |  Duty: {duty}/4095 ({duty/4095*100:.1f}%)")
    bus.write_i2c_block_data(
        PCA_ADDR,
        reg,
        [0x00, 0x00, duty & 0xFF, duty >> 8]
    )


# Init PWM frequency
set_pwm_freq(800)  # 50 Hz voor servo

motorMapping = [
    (8, 9),   # Motor 0: kanaal 2 en 3 (links voor)
    (12, 13),     # Motor 1: kanaal 0 en 1 (rechts voor)
    (10, 11),   # Motor 2: kanaal 6 en 7 (links achter)
    (14, 15)   # Motor 3: kanaal 4 en 5 (rechts achter)
]

def set_motor(motor, procent):
    # motor 0-3, procent -100 tot 100
    if motor < 0 or motor > 3:
        raise ValueError("Motor index must be between 0 and 3")
    if procent < -100 or procent > 100:
        raise ValueError("Procent must be between -100 and 100")

    # Converteer procent naar PWM duty cycle (0-4095)

    #als negatief kanaal 0 pwm ander kanaal 1 pwm procent omrekenen naar 0-4095
    if procent < 0:
        duty = int(-procent / 100 * 4095)
        set_pwm(motorMapping[motor][1], duty)  # kanaal 1, 3, 5, 7
        set_pwm(motorMapping[motor][0], 0)           # kanaal 0, 2, 4, 6 uit
    else:
        duty = int(procent / 100 * 4095)
        set_pwm(motorMapping[motor][0], duty)       # kanaal 0, 2, 4, 6
        set_pwm(motorMapping[motor][1], 0)     # kanaal 1, 3, 5, 7 uit

def rem_motor(duration):
    if motor < 0 or motor > 3:
        raise ValueError("Motor index must be between 0 and 3")
        
    set_pwm(motorMapping[motor][1], 4095)
    set_pwm(motorMapping[motor][0], 4095)


# Test kanaal 0
if __name__ == "__main__":
    while True:

        set_motor(0, 100)  # Motor 0 vooruit met 50% snelheid
        set_motor(1, 100)  # Motor 1 vooruit met 50% snelheid
        set_motor(2, 100)  # Motor 2 vooruit met 50% snelheid
        set_motor(3, 100)  # Motor 3 vooruit met 50% snelheid
        
        time.sleep(5)

        set_motor(0, 0)  # Motor 0 vooruit met 50% snelheid
        set_motor(1, 0)  # Motor 1 vooruit met 50% snelheid
        set_motor(2, 0)  # Motor 2 vooruit met 50% snelheid
        set_motor(3, 0)  # Motor 3 vooruit met 50% snelheid

        time.sleep(2)