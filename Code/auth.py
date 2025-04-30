# Ask your mentor for your password and group ID
import socket


class Auth:
    def __init__(self):
        # You need to adapt these two parameter
        self.group_id = 0      # Should be 1,2,3, or 4
        self.password = "pwd"
        
        # Do not adapt these two parameters unless you are specifically asked to
        self.host     = "10.50.250.65"
        self.port     = 5543

        if self.group_id == 0:
            print("Ask your mentor for your group_id and password and write them into 'auth.py'")
            
        if self.is_port_open():
            print("Host/Port open and accessable")
        else:
            print("Host/Port closed or you are not in the VPN of UChicago. Doublecheck your VPN connection")
            
    def is_port_open(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            timeout = 2.0
            sock.settimeout(timeout)
            try:
                sock.connect((self.host,self.port))
                return True
            except:
                return False
