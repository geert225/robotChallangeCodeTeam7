import time

class PCA9685:
    # Registers
    MODE1     = 0x00
    PRESCALE  = 0xFE
    LED0_ON_L = 0x06

    def __init__(self, i2c_master, address=0x60):
        self.i2c = i2c_master
        self.address = address

    # ------------------------
    # Low level write + verify
    # ------------------------
    def write(self, reg, val):
        self.i2c.write_byte_data(self.address, reg, val)
        time.sleep(0.005)

        read_val = self.i2c.read_byte_data(self.address, reg)
        if read_val != val:
            print(f"Warning: reg 0x{reg:02X} wrote 0x{val:02X} but read 0x{read_val:02X}")
        else:
            print(f"OK: reg 0x{reg:02X} = 0x{read_val:02X}")

        return read_val

    # ------------------------
    # PWM frequentie instellen
    # ------------------------
    def set_pwm_freq(self, freq_hz):
        osc_hz = 25_000_000
        prescale = round(osc_hz / (4096 * freq_hz)) - 1

        if not 3 <= prescale <= 255:
            raise ValueError(f"Invalid prescale value: {prescale}")

        actual_freq = osc_hz / (4096 * (prescale + 1))
        freq_error_pct = abs(actual_freq - freq_hz) / freq_hz * 100

        mode1_init = 0x21  # AI + ALLCALL

        self.write(self.MODE1, mode1_init)
        self.write(self.MODE1, mode1_init | 0x10)  # sleep
        self.write(self.PRESCALE, prescale)
        self.write(self.MODE1, mode1_init)

        time.sleep(0.005)

        self.write(self.MODE1, mode1_init | 0x80)  # restart

        mode1 = self.i2c.read_byte_data(self.address, self.MODE1)

        print("MODE1:", bin(mode1))
        print(f"Prescale: {prescale} | Requested: {freq_hz} Hz | Actual: {actual_freq:.1f} Hz ({freq_error_pct:.1f}% off)")

        if freq_error_pct > 3:
            print("WARNING: PWM frequency error > 3%")

    # ------------------------
    # PWM direct (0-4095)
    # ------------------------
    def set_pwm(self, channel, duty):
        if not 0 <= channel <= 15:
            raise ValueError("Channel must be 0-15")

        duty = max(0, min(4095, duty))
        reg = self.LED0_ON_L + 4 * channel

        self.i2c.write_block_data(
            self.address,
            reg,
            [0x00, 0x00, duty & 0xFF, duty >> 8]
        )

    # ------------------------
    # PWM percentage (0-100%)
    # ------------------------
    def set_pwm_percent(self, channel, percent):
        if not 0 <= percent <= 100:
            raise ValueError("Percent must be 0-100")

        duty = int(4095 * (percent / 100.0))
        self.set_pwm(channel, duty)