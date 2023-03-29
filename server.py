import socket

class UDP_socket:
    def __init__(self, ip, port):
        self.IP = ip
        self.PORT = port
        self.ADDRESS = (ip, port)
        self.socketOpen()

    def socketOpen(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print('UDP Socket Connected[', self.IP, self.PORT, ']')

    def socketClose(self):
        self.sock.close()
        print('UDP Socket Closed[', self.IP, self.PORT, ']')

    def recv(self, buf_size):
        return self.sock.recvfrom(buf_size)

    def sendto(self, text):
        self.sock.sendto(text, self.ADDRESS)