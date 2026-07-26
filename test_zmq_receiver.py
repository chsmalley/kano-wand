import zmq

PORT = 5555

context = zmq.Context()

subscriber = context.socket(zmq.SUB)
subscriber.bind(f"tcp://*:{PORT}")
subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

print("Waiting for messages...")

while True:
    message = subscriber.recv_string()
    print("Received:", message)