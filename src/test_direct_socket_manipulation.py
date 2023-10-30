import socket
import struct
import json


def varint(n):
    """Encode a number as VarInt"""
    buf = b""
    while True:
        towrite = n & 0x7F
        n >>= 7
        if n:
            buf += struct.pack("B", towrite | 0x80)
        else:
            buf += struct.pack("B", towrite)
            break
    return buf


def send_packet(sock, packet_id, data):
    """Send a packet to the server"""
    packet = varint(packet_id) + data
    length = varint(len(packet))
    sock.sendall(length + packet)


def recv_packet(sock):
    """Receive a packet from the server"""
    data = sock.recv(1)
    if len(data) == 0:
        print("Connection closed by server or no data received.")
        return None, None
    packet_id = ord(data)
    data = sock.recv(4096)  # Assuming a max buffer size, adjust as needed
    return packet_id, data


def get_status(host, port):
    """Get the status of a Minecraft server"""
    sock = socket.create_connection((host, port))

    # Handshake
    send_packet(sock,
                0x00,
                varint(404) +
                host.encode("utf8") +
                varint(port) +
                b'\x01'
                )

    # Status request
    send_packet(sock, 0x00, b'')

    # Status response
    packet_id, data = recv_packet(sock)
    if packet_id == 0x00:
        # length = ord(data[0])
        json_data = json.loads(data[1:].decode("utf8"))
        print("Server Status:", json_data)
    else:
        print("Unexpected packet:", packet_id)

    # Close the socket
    sock.close()


# Replace 'localhost' and 25565 with your server's IP and port
get_status('127.0.0.1', 25565)
