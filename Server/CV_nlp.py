from socket import socket
from socket import  AF_INET
from socket import SOCK_STREAM
from socket import SOL_SOCKET
from socket import SO_REUSEADDR
from threading import Thread

nlp_client = None
blender_client = None

# def init_CV_NLP():
#     # assigning default values to coordinates
#
#
#     # establishing connection
#     SERVER = "196.194.235.248"
#     PORT = 10005
#     client = socket(AF_INET, SOCK_STREAM)
#     client.connect((SERVER, PORT))
#     return client


def init_server():
    # LOCALHOST = '127.0.0.1'
    LOCALHOST = '192.168.100.3'
    PORT = 10005
    #PORT_blender_cv = 10010
    server = socket(AF_INET, SOCK_STREAM)
    server.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    server.bind((LOCALHOST, PORT))
    global nlp_client
    global blender_client
    #server_cv_blender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #server_cv_blender.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #server_cv_blender.bind((LOCALHOST, PORT_blender_cv))
    #print("Server started")
    # creating threads for every client
    #special_listen = Thread(target=listen_special, args=(server_cv_blender,))
    #special_listen.start()

    for i in range(2):

        print("Waiting for client request..")
        server.listen(1)
        clientsock, clientAddress = server.accept()
        input_client = clientsock.recv(1024)
        print(input_client)
        if input_client.decode().upper() == 'NLP':
            nlp_client = clientsock
            print('got NLP connection')
            nlp_client.sendall(bytes('connected ya beatches', 'UTF-8'))
        elif input_client.decode().upper() == 'BLENDER':
            blender_client = clientsock
            print('got blender connection')
            blender_client.sendall(bytes('connected ya beatches', 'UTF-8'))
        else:
            print('invalid Connection string')
            clientsock.sendall(bytes('invalid Connection string', 'UTF-8'))

def communicate():
    while True:
        user_input = input("input nlp or blender \n")
        if user_input == 'nlp':

            user_input = input("Enter the information that you wish to send")
            nlp_client.sendall(bytes('cv_input', 'UTF-8'))
            nlp_client.sendall(bytes(user_input, 'UTF-8'))
        elif user_input =='blender':
            # client.sendall(bytes(user_input, 'UTF-8'))
            # server_message = client.recv(2056)
            # print(server_message)
            # if server_message == bytes('got it', 'UTF-8'):
            user_input  = input("Enter the state ")
            blender_client.sendall(bytes('update_state', 'UTF-8'))
            print(int(user_input).to_bytes(1, byteorder='big'))
            blender_client.send(int(user_input).to_bytes(1, byteorder='big'))
        else:
            print('Wrong input ')


if __name__ == '__main__':
    # client = init_CV_NLP()
    # client.sendall(bytes('CV_NLP', 'UTF-8'))
    # status = client.recv(1024)
    # print(status.decode())
    networking = Thread(target=communicate, args=())
    networking.start()
    server = Thread(target=init_server(), args=())
    server.start()


    # do some other work