# Code for CV Client which continuously sends data to server by using keywords like GLOBAL and USERINPUT
import socket
import threading
import random
import time

client = None
startit = False
# global variables
coord_one = 0
coord_two = 0
user_input = ''


# def init_CV():
#     # assigning default values to coordinates
#
#
#     # establishing connection
#     SERVER = "196.194.235.248"
#     PORT = 10005
#     client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     client.connect((SERVER, PORT))
#     return client
def init_server():
    #LOCALHOST = '127.0.0.1'
    LOCALHOST = '192.168.100.3'
    PORT = 10010
    global startit
    # PORT_blender_cv = 10010
    # server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # server.bind((LOCALHOST, PORT))
    # global nlp_client
    global client
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LOCALHOST, PORT))
    # print("Server started")
    # creating threads for every client
    # special_listen = Thread(target=listen_special, args=(server_cv_blender,))
    # special_listen.start()


    print("Waiting for client request..")
    server.listen(1)
    clientsock, clientAddress = server.accept()
    client = clientsock
    print('got connection from {}'.format(clientAddress))
    startit = True
    print(startit)


def close_CV(client):
    client.close()


# class global_send(threading.Thread):
#     def __init__(self, client):
#         threading.Thread.__init__(self)
#         self.client = client
#     def run(self):
def global_send():
    global coord_one
    global coord_two
    global client
    global startit
    while not startit:
        a = 0
    while True:
        clientinput = "GLOBAL"
        precision = 10000
        #print('sending')
        # for translation multiply with precision and then add precision
        coord_one = 0#random.randint(0, (precision * 2) + 1)
        coord_two = 0#random.randint(0, (precision * 2) + 1)
        # Signally that this is the global data
        # client.sendall(bytes(clientinput, 'UTF-8'))
        # client.recv(2048).decode()
        # sending coord_one
        client.sendall((int(coord_one).to_bytes(4, byteorder='big')))
        client.recv(2048).decode()
        # sending coord two
        client.sendall((int(coord_one).to_bytes(4, byteorder='big')))
        client.recv(2048).decode()
        # send data after every 0.5 seconds
        time.sleep(0.1)



if __name__ == '__main__':
    # intialising client
    coords_send = threading.Thread(target=global_send, args=())
    server = threading.Thread(target=init_server, args=())
    server.start()
    coords_send.start()
    # making a thread for each task


    # do some other shiz
