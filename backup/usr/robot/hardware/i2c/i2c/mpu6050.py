import time


class MPU6050:
    # Registers
    PWR_MGMT_1   = 0x6B
    ACCEL_XOUT_H = 0x3B
    TEMP_OUT_H   = 0x41
    GYRO_XOUT_H  = 0x43

    def __init__(self, i2c_master, address=0x68):
        self.i2c = i2c_master
        self.address = address

        # Wake up device (clear sleep bit)
        self.i2c.write_byte_data(self.address, self.PWR_MGMT_1, 0x00)
        time.sleep(0.1)

    # ------------------------
    # Helper: combine bytes
    # ------------------------
    def _combine(self, high, low):
        value = (high << 8) | low
        if value >= 0x8000:
            value -= 65536
        return value

    # ------------------------
    # Raw reads
    # ------------------------
    def read_accel_raw(self):
        data = self.i2c.read_block_data(self.address, self.ACCEL_XOUT_H, 6)

        ax = self._combine(data[0], data[1])
        ay = self._combine(data[2], data[3])
        az = self._combine(data[4], data[5])

        return ax, ay, az

    def read_gyro_raw(self):
        data = self.i2c.read_block_data(self.address, self.GYRO_XOUT_H, 6)

        gx = self._combine(data[0], data[1])
        gy = self._combine(data[2], data[3])
        gz = self._combine(data[4], data[5])

        return gx, gy, gz

    def read_temp_raw(self):
        data = self.i2c.read_block_data(self.address, self.TEMP_OUT_H, 2)
        return self._combine(data[0], data[1])

    # ------------------------
    # Scaled values
    # ------------------------
    def read_accel_g(self):
        ax, ay, az = self.read_accel_raw()

        # default ±2g → 16384 LSB/g
        return ax / 16384.0, ay / 16384.0, az / 16384.0

    def read_gyro_dps(self):
        gx, gy, gz = self.read_gyro_raw()

        # default ±250°/s → 131 LSB/(°/s)
        return gx / 131.0, gy / 131.0, gz / 131.0

    def read_temp_c(self):
        temp_raw = self.read_temp_raw()
        return (temp_raw / 340.0) + 36.53

    # ------------------------
    # Alles tegelijk lezen
    # ------------------------
    def read_all(self):
        data = self.i2c.read_block_data(self.address, self.ACCEL_XOUT_H, 14)

        ax = self._combine(data[0], data[1])
        ay = self._combine(data[2], data[3])
        az = self._combine(data[4], data[5])

        temp = self._combine(data[6], data[7])

        gx = self._combine(data[8], data[9])
        gy = self._combine(data[10], data[11])
        gz = self._combine(data[12], data[13])

        return {
            "accel_raw": (ax, ay, az),
            "gyro_raw": (gx, gy, gz),
            "temp_raw": temp,
            "accel_g": (ax / 16384.0, ay / 16384.0, az / 16384.0),
            "gyro_dps": (gx / 131.0, gy / 131.0, gz / 131.0),
            "temp_c": (temp / 340.0) + 36.53
        }