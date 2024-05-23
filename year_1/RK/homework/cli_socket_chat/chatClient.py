import socket
import struct
import sys
import threading
import json
from datetime import datetime
import time
from enum import Enum

PORT = 1234 
HEADER_LENGTH = 2


class MessageType(Enum):
    PUBLIC = 1
    PRIVATE = 2
    USER_ANNOUNCE = 3
    ERROR = 4


def receive_fixed_length_msg(sock, msglen):
    message = b''
    while len(message) < msglen:
        chunk = sock.recv(msglen - len(message))  # preberi nekaj bajtov
        if chunk == b'':
            raise RuntimeError("socket connection broken")
        message = message + chunk  # pripni prebrane bajte sporocilu

    return message


def receive_message(sock):
    # preberi glavo sporocila (v prvih 2 bytih je dolzina sporocila)
    header = receive_fixed_length_msg(sock, HEADER_LENGTH)
    # pretvori dolzino sporocila v int
    message_length = struct.unpack("!H", header)[0]

    message = None
    if message_length > 0:  # ce je vse OK
        message = receive_fixed_length_msg(
            sock, message_length)  # preberi sporocilo
        message = message.decode("utf-8")

    return message


def send_message(sock, message):
    # pretvori sporocilo v niz bajtov, uporabi UTF-8 kodno tabelo
    encoded_message = message.encode("utf-8")

    # ustvari glavo v prvih 2 bytih je dolzina sporocila (HEADER_LENGTH)
    # metoda pack "!H" : !=network byte order, H=unsigned short
    header = struct.pack("!H", len(encoded_message))

    # najprj posljemo dolzino sporocilo, slee nato sporocilo samo
    message = header + encoded_message
    sock.sendall(message)


# message_receiver funkcija tece v loceni niti
def message_receiver():
    while True:
        msg = json.loads(receive_message(sock))

        sent_at = datetime.fromtimestamp(
            int(msg["sent_at"])).strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{sent_at} [{msg['sender']}] {msg['content']}")


# povezi se na streznik
print("[system] connecting to chat server ...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("localhost", PORT))
print("[system] connected!")

# zazeni message_receiver funkcijo v loceni niti
thread = threading.Thread(target=message_receiver)
thread.daemon = True
thread.start()

username = input("Enter username: ")

while username == "" or username == "RKchat":
    if username == "":
        print("Username cannot be empty")
    else:
        print('Username cannot be the same name as the server username')
    username = input("Enter username: ")

# announce username
send_message(sock, json.dumps({
    "type": MessageType.USER_ANNOUNCE.value,
    "username": username,
    "sent_at": int(time.time())
}))

# pocakaj da uporabnik nekaj natipka in poslji na streznik
while True:
    try:
        message_object = {}

        type = ""

        while type != "public" and type != "private":
            type = input(
                "Enter message type (public | private): ")

            if type == "public":
                message_object['type'] = MessageType.PUBLIC.value
            elif type == "private":
                message_object['type'] = MessageType.PRIVATE.value

        if message_object['type'] == MessageType.PRIVATE.value:
            message_object['recipient'] = input("Enter recipient username: ")

        message_object['content'] = input("Enter message content: ")
        message_object['sent_at'] = int(time.time())
        send_message(sock, json.dumps(message_object))
    except KeyboardInterrupt:
        sys.exit()
