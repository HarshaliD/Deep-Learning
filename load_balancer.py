
import time

# Simulate a server that processes requests
class Server:
    def __init__(self, id):
        self.id = id

    def handle_request(self, request):
        print(f"Server {self.id} is processing {request}")

# Simple load balancer that uses round-robin to distribute requests
class LoadBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.index = 0  # To track which server to use

    def distribute_request(self, request):
        server = self.servers[self.index]
        server.handle_request(request)
        self.index = (self.index + 1) % len(self.servers)  # Move to the next server

# Create a few servers
servers = [Server(1), Server(2), Server(3)]

# Create load balancer
lb = LoadBalancer(servers)

# Simulate requests
for i in range(5):
    lb.distribute_request(f"Request {i + 1}")
    time.sleep(0.5)  # Simulate a small delay between requests
