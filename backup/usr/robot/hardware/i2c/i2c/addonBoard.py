import time


class AddonBoard:
    # I2C address (moet matchen met Arduino)
    DEFAULT_ADDRESS = 0x08

    # Commands (zoals in Arduino code)
    CMD_SERVO = 0x01
    CMD_LED   = 0x02
    CMD_ULTRA = 0x03

    def __init__(self, i2c_master, address=DEFAULT_ADDRESS):
        self.i2c = i2c_master
        self.address = address

    # ------------------------
    # Low-level command sender
    # ------------------------
    def _send_command(self, cmd, data=None):
        if data is None:
            data = []

        if not isinstance(data, list):
            raise TypeError("data must be a list")

        payload = [cmd] + data
        self.i2c.write_raw(self.address, payload)

    # ------------------------
    # Servo control
    # ------------------------
    def set_servo(self, angle1, angle2):
        """
        angle1, angle2: 0 - 180 graden
        """
        angle1 = max(0, min(180, int(angle1)))
        angle2 = max(0, min(180, int(angle2)))

        self._send_command(self.CMD_SERVO, [angle1, angle2])

    # ------------------------
    # LED control
    # ------------------------
    def set_led(self, r, g, b):
        """
        RGB: 0 - 255
        """
        r = max(0, min(255, int(r)))
        g = max(0, min(255, int(g)))
        b = max(0, min(255, int(b)))

        self._send_command(self.CMD_LED, [r, g, b])

    # ------------------------
    # Ultrasonic read
    # ------------------------
    def read_ultrasonic(self):
        """
        Returns:
            (dist1_cm, dist2_cm)
        """
        # 1. stuur request command
        self._send_command(self.CMD_ULTRA)

        # kleine delay nodig omdat Arduino async verwerkt
        time.sleep(0.01)

        # 2. lees 4 bytes terug
        data = self.i2c.read_raw(self.address, 4)

        if len(data) != 4:
            raise RuntimeError("Invalid ultrasonic response")

        dist1 = (data[0] << 8) | data[1]
        dist2 = (data[2] << 8) | data[3]

        return dist1, dist2