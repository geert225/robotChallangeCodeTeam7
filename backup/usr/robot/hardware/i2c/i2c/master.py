from smbus2 import SMBus, i2c_msg
import time

class I2CMaster:
    def __init__(self, bus_id=1, retries=3, delay=0.05):
        self.bus_id = bus_id
        self.bus = SMBus(bus_id)
        self.retries = retries
        self.delay = delay

    def _execute(self, func, *args):
        """Interne wrapper met retry + foutafhandeling"""
        for attempt in range(self.retries):
            try:
                return func(*args)
            except OSError as e:
                if e.errno == 121:
                    print(f"I2C fout (Errno 121), poging {attempt+1}/{self.retries}")
                    time.sleep(self.delay)
                else:
                    raise
        print("I2C device niet bereikbaar na retries")
        return None  # of raise als je harder wilt falen

    def write_byte(self, addr, value):
        self._execute(self.bus.write_byte, addr, value)

    def write_byte_data(self, addr, reg, value):
        self._execute(self.bus.write_byte_data, addr, reg, value)

    def write_block_data(self, addr, reg, data):
        if not isinstance(data, list):
            raise TypeError("data moet een list zijn")
        self._execute(self.bus.write_i2c_block_data, addr, reg, data)

    def read_byte(self, addr):
        return self._execute(self.bus.read_byte, addr)

    def read_byte_data(self, addr, reg):
        return self._execute(self.bus.read_byte_data, addr, reg)

    def read_block_data(self, addr, reg, length):
        return self._execute(self.bus.read_i2c_block_data, addr, reg, length)

    def read_raw(self, addr, length):
        def _read():
            msg = i2c_msg.read(addr, length)
            self.bus.i2c_rdwr(msg)
            return list(msg)
        return self._execute(_read)

    def write_raw(self, addr, data):
        if not isinstance(data, list):
            raise TypeError("data moet een list zijn")

        def _write():
            msg = i2c_msg.write(addr, data)
            self.bus.i2c_rdwr(msg)

        self._execute(_write)

    def close(self):
        self.bus.close()