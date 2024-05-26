import signal
import socket
import struct
import threading
import json
from enum import Enum
import time
import ssl

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
        chunk = sock.recv(msglen - len(message))
        if chunk == b'':
            raise RuntimeError("socket connection broken")
        message += chunk
    return message


def receive_message(sock):
    # Read the message header (first 2 bytes indicate message length)
    header = receive_fixed_length_msg(sock, HEADER_LENGTH)
    message_length = struct.unpack("!H", header)[0]

    message = None
    if message_length > 0:
        message = receive_fixed_length_msg(
            sock, message_length)
        message = message.decode("utf-8")

    return message


def send_message(sock, message):
    encoded_message = message.encode("utf-8")
    header = struct.pack("!H", len(encoded_message))
    message = header + encoded_message
    sock.sendall(message)


def client_thread(client_sock, client_addr):
    global clients

    print("[system] connected with " +
          client_addr[0] + ":" + str(client_addr[1]))
    print("[system] we now have " + str(len(clients)) + " clients")

    try:
        while True:  # Infinite loop
            msg_object = json.loads(receive_message(client_sock))

            if not msg_object:  # If there's a message
                break

            sender = find_client_by_socket(client_sock)

            msg_object['sender'] = sender['common_name']

            if msg_object['type'] == MessageType.USER_ANNOUNCE.value:
                cert = client_sock.getpeercert()
                for subject in cert['subject']:
                    if subject[0][0] == 'commonName':
                        client_cn = subject[0][1]
                with clients_lock:
                    sender['common_name'] = client_cn

            if msg_object['type'] == MessageType.PUBLIC.value:
                with clients_lock:
                    for client in clients:
                        if client != sender:
                            send_message(client['socket'],
                                         json.dumps(msg_object))

            if msg_object['type'] == MessageType.PRIVATE.value:
                client = find_client_by_common_name(msg_object['recipient'])

                if client:
                    send_message(client['socket'], json.dumps(msg_object))
                else:
                    send_message(sender['socket'], json.dumps({
                        "sender": "RKchat",
                        "type": MessageType.ERROR.value,
                        "content": "No user with such common name found.",
                        "sent_at": int(time.time()),
                    }))

    except Exception as e:
        print(f"[error] {e}")

    with clients_lock:
        clients.remove(find_client_by_socket(client_sock))

    print("[system] we now have " + str(len(clients)) + " clients")
    client_sock.close()


def find_client_by_socket(client_sock):
    with clients_lock:
        for client in clients:
            if client["socket"] == client_sock:
                return client


def find_client_by_common_name(common_name):
    with clients_lock:
        for client in clients:
            if client["common_name"] == common_name:
                return client


# Create socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.set_ciphers('ECDHE-RSA-AES128-GCM-SHA256')
context.load_cert_chain(certfile='server.crt', keyfile='server.key')
context.load_verify_locations('ca.crt')
context.verify_mode = ssl.CERT_REQUIRED
server_socket.bind(("localhost", PORT))
server_socket.listen(1)

# Wait for new clients
print("[system] listening ...")
clients = list()
clients_lock = threading.Lock()
while True:
    try:
        # Wait for a new connection - blocking call
        client_sock, client_addr = server_socket.accept()

        secure_conn = context.wrap_socket(client_sock, server_side=True)

        client_cert = secure_conn.getpeercert()

        # Get common name
        for subject in client_cert['subject']:
            if subject[0][0] == 'commonName':
                client_cn = subject[0][1]

        with clients_lock:
            clients.append({
                "common_name": client_cn,
                "socket": secure_conn,
            })

        thread = threading.Thread(
            target=client_thread, args=(secure_conn, client_addr))
        thread.daemon = True
        thread.start()

    except KeyboardInterrupt:
        break

print("[system] closing server socket ...")
server_socket.close()
