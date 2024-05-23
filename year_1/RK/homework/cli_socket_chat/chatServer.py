import signal
import socket
import struct
import threading
import json
from enum import Enum
import time
from enum import Enum

signal.signal(signal.SIGINT, signal.SIG_DFL)

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


# funkcija za komunikacijo z odjemalcem (tece v loceni niti za vsakega odjemalca)
def client_thread(client_sock, client_addr):
    global clients

    print("[system] connected with " +
          client_addr[0] + ":" + str(client_addr[1]))
    print("[system] we now have " + str(len(clients)) + " clients")

    try:

        while True:  # neskoncna zanka
            msg_object = json.loads(receive_message(client_sock))

            if not msg_object:  # ce obstaja sporocilo
                break

            sender = find_client_by_socket(client_sock)

            msg_object['sender'] = sender['username']

            if msg_object['type'] == MessageType.USER_ANNOUNCE.value:
                with clients_lock:
                    sender['username'] = msg_object['username']

            if msg_object['type'] == MessageType.PUBLIC.value:
                for client in clients:
                    if client != sender:
                        send_message(client['socket'], json.dumps(msg_object))

            if msg_object['type'] == MessageType.PRIVATE.value:
                client = find_client_by_username(msg_object['recipient'])

                if client:
                    send_message(client['socket'], json.dumps(msg_object))
                else:
                    send_message(sender['socket'], json.dumps({
                        "sender": "RKchat",
                        "type": MessageType.ERROR.value,
                        "content": "No user with such username found.",
                        "sent_at": int(time.time()),
                    }))

    except:
        # tule bi lahko bolj elegantno reagirali, npr. na posamezne izjeme. Trenutno kar pozremo izjemo
        pass

    # prisli smo iz neskoncne zanke
    with clients_lock:
        clients.remove(find_client_by_socket(client_sock))

    print("[system] we now have " + str(len(clients)) + " clients")
    client_sock.close()


def find_client_by_socket(client_sock):
    for client in clients:
        if client["socket"] == client_sock:
            return client


def find_client_by_username(username):
    for client in clients:
        if client["username"] == username:
            return client


# kreiraj socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("localhost", PORT))
server_socket.listen(1)

# cakaj na nove odjemalce
print("[system] listening ...")
clients = list()
clients_lock = threading.Lock()
while True:
    try:
        # pocakaj na novo povezavo - blokirajoc klic
        client_sock, client_addr = server_socket.accept()
        with clients_lock:
            clients.append({
                "username": "",
                "socket": client_sock,
            })

        thread = threading.Thread(
            target=client_thread, args=(client_sock, client_addr))
        thread.daemon = True
        thread.start()

    except KeyboardInterrupt:
        break

print("[system] closing server socket ...")
server_socket.close()
