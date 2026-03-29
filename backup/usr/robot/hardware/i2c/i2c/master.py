from smbus2 import SMBus, i2c_msg

class I2CMaster:
    def __init__(self, bus_id=1):
        """
        bus_id:
            0 → oudere Raspberry Pi
            1 → moderne Raspberry Pi (meest gebruikt)
        """
        self.bus_id = bus_id
        self.bus = SMBus(bus_id)

    def write_byte(self, addr, value):
        """Schrijf 1 byte naar device"""
        self.bus.write_byte(addr, value)

    def write_byte_data(self, addr, reg, value):
        """Schrijf 1 byte naar register"""
        self.bus.write_byte_data(addr, reg, value)

    def write_block_data(self, addr, reg, data):
        """Schrijf meerdere bytes naar register"""
        if not isinstance(data, list):
            raise TypeError("data moet een list zijn")
        self.bus.write_i2c_block_data(addr, reg, data)

    def read_byte(self, addr):
        """Lees 1 byte"""
        return self.bus.read_byte(addr)

    def read_byte_data(self, addr, reg):
        """Lees 1 byte van register"""
        return self.bus.read_byte_data(addr, reg)

    def read_block_data(self, addr, reg, length):
        """Lees meerdere bytes van register"""
        return self.bus.read_i2c_block_data(addr, reg, length)

    def read_raw(self, addr, length):
        """
        Lees raw data zonder register (lage-level control)
        """
        msg = i2c_msg.read(addr, length)
        self.bus.i2c_rdwr(msg)
        return list(msg)

    def write_raw(self, addr, data):
        """
        Schrijf raw data zonder register
        """
        if not isinstance(data, list):
            raise TypeError("data moet een list zijn")
        msg = i2c_msg.write(addr, data)
        self.bus.i2c_rdwr(msg)

    def close(self):
        """Sluit de bus"""
        self.bus.close()