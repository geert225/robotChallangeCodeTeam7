import os
import mmap
import struct
import time
import fcntl

from motor import set_rad, rem_rad
import time

ROTATIES = 3
ENC_PPR = 1238 #1441

num_encoders = 5

shm_file_path = "/dev/shm/encoder_positions"
fd = os.open(shm_file_path, os.O_CREAT | os.O_RDWR)
os.ftruncate(fd, num_encoders*8)
shared_mem = mmap.mmap(fd, num_encoders*8, mmap.MAP_SHARED, mmap.PROT_READ)
os.close(fd)

def read_encoder():
    offset = 4*8
    data = shared_mem[offset:offset+8]
    value = struct.unpack("q", data)[0]
    return value


def main():
    start_enc = read_encoder()
    end_enc = start_enc + (ROTATIES * ENC_PPR)

    set_rad(75)

    while True:
        slow = False
        act_enc = read_encoder()

        if ((end_enc - act_enc) < ((ENC_PPR / 3) * 2)) and (slow == False):
            slow = True
            set_rad(30)

        if ((end_enc - act_enc) < 10):
            #set_rad(10)
            #rem_rad()
            break

        time.sleep(0.01)



    rem_rad()
    time.sleep(1)
    set_rad(0)

    act_enc = read_encoder()
    print(f"encode waarde {end_enc - act_enc}")


while True:
    main()
    time.sleep(1)