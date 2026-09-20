import os
import cv2
import threading
import json
import asyncio
import sys
from collections import OrderedDict
import picamera
import subprocess
import zmq

from aiohttp import web
# from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc import RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from nerf_shooter import main as nerf_main
from pitrack import H264EncodedStreamTrack
from aiortc.rtcrtpparameters import RTCRtpCodecCapability
from rtcpeerconnection import RTCPeerConnection
# from rtcrtpsender import RTCRtpSender


FRAME_RATE = 30
CAMERA_RESOLUTION = (640, 480)
BASE_PATH = os.path.dirname(__file__)
ZMQ_PORT = 5555

camera = None

codec_parameters = OrderedDict(
    [
        ("packetization-mode", "1"),
        ("level-asymmetry-allowed", "1"),
        ("profile-level-id", "42001f"),
    ]
)
pi_capability = RTCRtpCodecCapability(
    mimeType="video/H264", clockRate=90000, channels=None, parameters=codec_parameters
)
preferences = [pi_capability]
pcs = set()

def zmq_listener():
    global latest_message
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.bind(f"tcp://*:{ZMQ_PORT}")
    # Subscribe to everything
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    print("ZeroMQ listener started...")
    while True:
        message = socket.recv_string()
        print("Received:", message)
        try:
            data = json.loads(message)
            print(data)
        except (TypeError, json.JSONDecodeError):
            print("Ignoring invalid Nerf command")
            continue

        if data.get("command") != "shoot":
            print("Ignoring non-shoot Nerf command")
            continue

        latest_message = message
        subprocess.Popen([
            sys.executable,
            os.path.join(BASE_PATH, "nerf_shooter.py"),
        ])


class CameraStreamTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        
    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return VideoFrame.from_ndarray(frame, format="rgb24")

async def index(request):
    content = open('index.html', 'r').read()
    return web.Response(content_type='text/html', text=content)

async def javascript(request):
    content = open(os.path.join(BASE_PATH, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def shoot(request):
    print("running shoot")
    script_thread = threading.Thread(target=nerf_main)
    script_thread.start()
    return web.json_response({"status": "shooting nerf dart..."})

async def offer(request):
    global camera
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    video_track = H264EncodedStreamTrack(FRAME_RATE)
    if not camera:
        camera = picamera.PiCamera()
        camera.resolution = CAMERA_RESOLUTION
        camera.framerate = FRAME_RATE
    else:
        camera.stop_recording()

    camera.start_recording(
        video_track,
        format="h264",
        profile="constrained",
        inline_headers=True,
        sei=False,
    )

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print("ICE connection state is %s" % pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    for t in pc.getTransceivers():
        if t.kind == "video" and video_track:
            t.setCodecPreferences(preferences)
            pc.addTrack(video_track)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def on_shutdown(app):
    global camera
    # close peer connections
    print("Shutting down")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    camera.stop_recording()
    camera.close()


if __name__ == "__main__":

    listener = threading.Thread(
        target=zmq_listener,
        daemon=True
    )
    listener.start()

    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_get("/client.js", javascript)
    app.router.add_post('/offer', offer)
    app.router.add_post('/shoot', shoot)

    web.run_app(app, port=8080)
