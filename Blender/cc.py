import socket
import mmap
import time
#from multiprocessing import shared_memory


if __name__ == '__main__':
    # establishing connection
    SERVER = '43.245.206.104'#'43.251.253.156'#'43.245.206.104'#'196.194.235.248'#"43.245.206.104"
    PORT = 10010
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((SERVER, PORT))
    precision = 10000
    #shared_data = shared_memory.ShareableList(name='coords')
    set = 1
    while True:
        #time.sleep(0.2)
        x = client.recv(2048)
        client.sendall(bytes('got x', 'UTF-8'))
        y = client.recv(2048)
        client.sendall(bytes('got y ', 'UTF-8'))
        print(x)
        print(y)
        with open("D:\\BL8\\communicate.txt", "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            mm.write(set.to_bytes(1, byteorder='big'))
            mm.write(x)
            mm.write(y)

            # print(int.from_bytes(mm[1:5], byteorder='big'))
            # print(int.from_bytes(x, byteorder='big'))
            f.close()