#!/usr/bin/env python3

import asyncio
import json
import math
from typing import List, Dict, Any, Optional

import cv2
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

mp_hands = mp.solutions.hands

THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0

app = FastAPI()

# Serve static files (index.html, css, js)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    # Serve our static index page
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast(self, message: str):
        to_remove = []
        for ws in list(self.active):
            try:
                await ws.send_text(message)
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            self.disconnect(ws)


manager = ConnectionManager()


class HandState:
    def __init__(self):
        self.pinch_open_ref: Optional[Dict[str, float]] = None
        self.pinch_closed_ref: Optional[Dict[str, float]] = None
        self.pinch_threshold: float = 0.5

        self.last_metrics: Optional[Dict[str, float]] = None
        self.last_hand: Optional[str] = None


state = HandState()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for two-way communication:
    - Server -> client: streaming hand data.
    - Client -> server: calibration commands, threshold updates.
    """
    await manager.connect(websocket)
    try:
        while True:
            msg_text = await websocket.receive_text()
            try:
                msg = json.loads(msg_text)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "set_open":
                if state.last_metrics is not None:
                    state.pinch_open_ref = dict(state.last_metrics)
                    print("Captured pinch OPEN reference:", state.pinch_open_ref)
            elif msg_type == "set_closed":
                if state.last_metrics is not None:
                    state.pinch_closed_ref = dict(state.last_metrics)
                    print("Captured pinch CLOSED reference:", state.pinch_closed_ref)
            elif msg_type == "set_threshold":
                value = msg.get("value")
                try:
                    value = float(value)
                    state.pinch_threshold = max(0.0, min(1.0, value))
                    print("Set pinch threshold:", state.pinch_threshold)
                except (TypeError, ValueError):
                    pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)


def compute_pinch_metrics(lms) -> Dict[str, float]:
    thumb = lms[THUMB_TIP]
    index = lms[INDEX_TIP]
    wrist = lms[WRIST]

    # Distance between thumb-tip and index-tip in world coords (meters)
    dx = thumb.x - index.x
    dy = thumb.y - index.y
    dz = thumb.z - index.z
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)

    # Angle between wrist->thumb and wrist->index
    v1 = (thumb.x - wrist.x, thumb.y - wrist.y, thumb.z - wrist.z)
    v2 = (index.x - wrist.x, index.y - wrist.y, index.z - wrist.z)

    def norm(v):
        return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    n1 = norm(v1)
    n2 = norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        angle_deg = 0.0
    else:
        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        cosang = max(min(dot / (n1 * n2), 1.0), -1.0)
        angle_deg = math.degrees(math.acos(cosang))

    return {"distance": dist, "angle_deg": angle_deg}


async def hand_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera 0")
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                await asyncio.sleep(0.01)
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            payload: Dict[str, Any] = {
                "hand": None,
                "wrist": None,
                "pinch": None,
            }

            if results.multi_hand_world_landmarks and results.multi_handedness:
                # Prefer right hand if present
                labels = [h.classification[0].label.lower()
                          for h in results.multi_handedness]
                chosen_idx = 0
                if "right" in labels:
                    chosen_idx = labels.index("right")

                lms = results.multi_hand_world_landmarks[chosen_idx].landmark
                hand_label = labels[chosen_idx]

                # Wrist pose
                wrist_lm = lms[WRIST]
                wrist = {
                    "x": wrist_lm.x,
                    "y": wrist_lm.y,
                    "z": wrist_lm.z,
                }

                # Pinch metrics
                pinch_raw = compute_pinch_metrics(lms)
                state.last_metrics = pinch_raw
                state.last_hand = hand_label

                progress = None
                pinch_state = "UNCALIBRATED"

                calibrated = state.pinch_open_ref is not None and state.pinch_closed_ref is not None
                if calibrated:
                    open_d = state.pinch_open_ref["distance"]
                    closed_d = state.pinch_closed_ref["distance"]
                    denom = max(abs(open_d - closed_d), 1e-6)
                    progress = (open_d - pinch_raw["distance"]) / denom
                    progress = max(0.0, min(1.0, progress))
                    pinch_state = "CLOSED" if progress >= state.pinch_threshold else "OPEN"

                payload = {
                    "hand": hand_label,
                    "wrist": wrist,
                    "pinch": {
                        "distance": pinch_raw["distance"],
                        "angle_deg": pinch_raw["angle_deg"],
                        "progress": progress,
                        "state": pinch_state,
                        "threshold": state.pinch_threshold,
                        "calibrated": calibrated,
                    },
                }
            else:
                # No hand detected
                state.last_metrics = None
                state.last_hand = None

            await manager.broadcast(json.dumps(payload))
            await asyncio.sleep(0.02)  # ~50 Hz
    finally:
        cap.release()
        hands.close()


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(hand_loop())


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
