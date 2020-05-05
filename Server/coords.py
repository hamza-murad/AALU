import socket
import mmap
#from multiprocessing import shared_memory


if __name__ == '__main__':
    # establishing connection
    SERVER = "127.0.0.1"
    PORT = 10010
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((SERVER, PORT))
    print('connection establishded')
    precision = 10000
    #shared_data = shared_memory.ShareableList(name='coords')
    set = 1
    while True:
        x = client.recv(2048)
        client.sendall(bytes('got x', 'UTF-8'))
        y = client.recv(2048)
        client.sendall(bytes('got y ', 'UTF-8'))

        with open("communicate.txt", "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            mm.write(set.to_bytes(1, byteorder='big'))
            mm.write(x)
            mm.write(y)

            print(int.from_bytes(mm[1:5], byteorder='big'))
            # print(int.from_bytes(x, byteorder='big'))
            f.close()

