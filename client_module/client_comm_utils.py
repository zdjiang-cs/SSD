import os
import sys
import struct
import socket
import pickle
from time import sleep
import time

def killport(port):
    command = '''kill -9 $(netstat -nlp | grep :''' + str(
        port) + ''' | awk '{print $7}' | awk -F"/" '{ print $1 }')'''
    os.system(command)

def connect_send_socket(dst_ip, dst_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    while s.connect_ex((dst_ip, dst_port)) != 0:
        sleep(0.5)

    return s

def connect_get_socket(listen_ip, listen_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    start_time = time.time()
    while True:
        try:
            s.bind((listen_ip, listen_port))
            break
        except OSError as e:
            print(e)
            print("**OSError**", listen_ip, listen_port)
            sleep(0.7)
            if time.time() - start_time > 15:
                sys.exit(0)
    s.listen(1)

    conn, _ = s.accept()
    # conn.settimeout(600)

    return conn

def send_data_socket(data, s):
    data = pickle.dumps(data)
    s.sendall(struct.pack(">I", len(data)))
    s.sendall(data)

def get_data_socket(conn):
    data_len = struct.unpack(">I", conn.recv(4))[0]
    data = conn.recv(data_len, socket.MSG_WAITALL)
    recv_data = pickle.loads(data)

    return recv_data
