import socket
import struct
import sys
import threading
import json
from datetime import datetime
import time
from enum import Enum
import ssl

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
    if message_length > 0:  # If everything is OK
        message = receive_fixed_length_msg(
            sock, message_length)  # Read the message
        message = message.decode("utf-8")

    return message


def send_message(sock, message):
    encoded_message = message.encode("utf-8")
    header = struct.pack("!H", len(encoded_message))
    message = header + encoded_message
    sock.sendall(message)


def message_receiver():
    while True:
        msg = json.loads(receive_message(sock))

        sent_at = datetime.fromtimestamp(
            int(msg["sent_at"])).strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{sent_at} [{msg['sender']}] {msg['content']}")


print("[system] connecting to chat server ...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
context.set_ciphers('ECDHE-RSA-AES128-GCM-SHA256')

cert = input("Enter certificate name: ")
while cert == "":
    print("Cert cannot be empty")
    cert = input("Enter certificate name: ")

context.load_cert_chain(certfile=f"{cert}.crt", keyfile=f"{cert}.key")
context.load_verify_locations('ca.crt')
sock = context.wrap_socket(s, server_hostname='localhost')
sock.connect(("localhost", PORT))
print("[system] connected!")

# Start the message_receiver function in a separate thread
thread = threading.Thread(target=message_receiver)
thread.daemon = True
thread.start()

# Announce username
send_message(sock, json.dumps({
    "type": MessageType.USER_ANNOUNCE.value,
    "sent_at": int(time.time())
}))

while True:
    try:
        message_object = {}

        type = ""

        while type != "public" and type != "private":
            type = input("Enter message type (public | private): ")

            if type == "public":
                message_object['type'] = MessageType.PUBLIC.value
            elif type == "private":
                message_object['type'] = MessageType.PRIVATE.value

        if message_object['type'] == MessageType.PRIVATE.value:
            message_object['recipient'] = input(
                "Enter recipient common name: ")

        message_object['content'] = input("Enter message content: ")
        message_object['sent_at'] = int(time.time())
        send_message(sock, json.dumps(message_object))
    except KeyboardInterrupt:
        sys.exit()

