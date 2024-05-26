-- CA --
openssl genrsa -out ca.key 2048
openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 -out ca.crt -subj "/CN=CA"


-- SERVER --
openssl genrsa -out server.key 2048
openssl req -new -key server.key -out server.csr -subj "/CN=localhost"

openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365 -sha256

-- CLIENTS (client1, client2, client3)--
openssl genrsa -out client1.key 2048
openssl req -new -key client1.key -out client1.csr -subj "/CN=client1"

openssl x509 -req -in client1.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client1.crt -days 365 -sha256


openssl genrsa -out client2.key 2048
openssl req -new -key client2.key -out client2.csr -subj "/CN=client2"

openssl x509 -req -in client2.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client2.crt -days 365 -sha256


openssl genrsa -out client3.key 2048
openssl req -new -key client3.key -out client3.csr -subj "/CN=client3"

openssl x509 -req -in client3.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out client3.crt -days 365 -sha256

