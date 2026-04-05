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

        #print(f"sending payload: {payload}")
        self.i2c.write_raw(self.address, payload)

    # ------------------------
    # Servo control
    # ------------------------
    def set_servo(self, mode):
        #print(f"set servo {mode}")
        self._send_command(self.CMD_SERVO, [mode])

    # ------------------------
    # LED control
    # ------------------------
    def set_led(self, m, r1, g1, b1, r2, g2, b2):
        """
        RGB: 0 - 255
        """
        r1 = max(0, min(3, int(r1)))
        r1 = max(0, min(255, int(r1)))
        g1 = max(0, min(255, int(g1)))
        b1 = max(0, min(255, int(b1)))
        r2 = max(0, min(255, int(r2)))
        g2 = max(0, min(255, int(g2)))
        b2 = max(0, min(255, int(b2)))

        self._send_command(self.CMD_LED, [m, r1, g1, b1, r2, g2, b2])

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