/dev/shm/encoder_positions
int64 -> absolute encoder waarde links voor
int64 -> absolute encoder waarde rechts voor
int64 -> absolute encoder waarde links achter
int64 -> absolute encoder waarde rechts achter


/dev/shm/pwm_setpoints
uint16 -> pwm channel 0
uint16 -> pwm channel 1
uint16 -> pwm channel 2
uint16 -> pwm channel 3
uint16 -> pwm channel 4
uint16 -> pwm channel 5
uint16 -> pwm channel 6
uint16 -> pwm channel 7
uint16 -> pwm channel 8
uint16 -> pwm channel 9
uint16 -> pwm channel 10
uint16 -> pwm channel 11
uint16 -> pwm channel 12
uint16 -> pwm channel 13
uint16 -> pwm channel 14
uint16 -> pwm channel 15


/dev/shm/ultrasoon
double -> update rate (0 = no update)
double -> timestamp
uint16 -> distance links
uint16 -> distance rechts


/dev/shm/led_ctrl
uint8 -> mode (modus 0 -> uit, modus 1 -> color 1, modus 2 -> color 1 flash on/off, modus 3 -> color 1 / 2 flash)
uint8 -> red 1
uint8 -> green 1
uint8 -> blue 1
uint8 -> red 2
uint8 -> green 2
uint8 -> blue 2


/dev/shm/servo
uint8 -> angle 1
uint8 -> angle 2


/dev/shm/robot_cmd
double -> vx
double -> vy
double -> omega
double -> timestamp


/dev/shm/vision_frame
double -> timestamp
variable -> vision frame buffer


/dev/shm/