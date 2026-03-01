"""
WebSocket endpoint for real-time webcam frame inference.

WS /ws/webcam

Protocol (client → server):
  Binary message: raw JPEG bytes of a single frame.

Protocol (server → client):
  JSON message: { "label": "Real"|"Fake", "confidence": 0.97, "prob_real": 0.97 }

Usage example (JavaScript):
  const ws = new WebSocket("ws://localhost:8000/ws/webcam");
  ws.binaryType = "arraybuffer";

  // Send a frame (captured via canvas):
  ws.send(jpegBytes);

  // Receive prediction:
  ws.onmessage = (event) => console.log(JSON.parse(event.data));
"""

import json
import os
import sys

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

router = APIRouter()


@router.websocket("/ws/webcam")
async def webcam_stream(websocket: WebSocket):
    """
    Accept binary JPEG frames from client, run inference, send back JSON results.
    Runs inference every frame — clients should throttle sending rate for performance.
    """
    await websocket.accept()

    from api.main import get_model
    from inference.predict import get_transform, preprocess_image, predict
    import torch

    model, device = get_model()
    transform = get_transform()

    try:
        while True:
            # Receive raw JPEG bytes
            data = await websocket.receive_bytes()

            # Decode JPEG → numpy BGR
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_text(json.dumps({"error": "Invalid frame"}))
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = preprocess_image(frame_rgb, transform)

            with torch.no_grad():
                label, confidence = predict(model, tensor, device)
                logit = model(tensor.to(device))[0, 0].item()
                prob_real = torch.sigmoid(torch.tensor(logit)).item()

            await websocket.send_text(json.dumps({
                "label": label,
                "confidence": round(confidence, 4),
                "prob_real": round(prob_real, 4),
            }))

    except WebSocketDisconnect:
        pass
