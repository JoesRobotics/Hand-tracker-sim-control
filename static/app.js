const handEl = document.getElementById("hand");
const wristEl = document.getElementById("wrist");
const distEl = document.getElementById("dist");
const angEl = document.getElementById("ang");
const progTextEl = document.getElementById("prog-text");
const progBarEl = document.getElementById("prog-bar");
const pstateEl = document.getElementById("pstate");

const threshSlider = document.getElementById("threshold");
const threshValueEl = document.getElementById("thresh-value");
const btnOpen = document.getElementById("btn-open");
const btnClosed = document.getElementById("btn-closed");

const connDot = document.getElementById("conn-dot");
const connStatus = document.getElementById("conn-status");

let ws = null;

function setStatus(connected) {
  if (connected) {
    connDot.classList.add("status-dot--ok");
    connStatus.textContent = "Connected";
  } else {
    connDot.classList.remove("status-dot--ok");
    connStatus.textContent = "Disconnected";
  }
}

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    setStatus(true);
    sendThreshold();
  };

  ws.onclose = () => {
    setStatus(false);
    setTimeout(connectWS, 1000);
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data || "{}");

    if (!data.hand) {
      handEl.textContent = "none";
      wristEl.textContent = "(-, -, -)";
      distEl.textContent = "-";
      angEl.textContent = "-";
      progTextEl.textContent = "-";
      progBarEl.style.width = "0%";
      pstateEl.textContent = "NO HAND";
      return;
    }

    handEl.textContent = data.hand;

    if (data.wrist) {
      const w = data.wrist;
      wristEl.textContent = `(${w.x.toFixed(3)}, ${w.y.toFixed(3)}, ${w.z.toFixed(3)})`;
    }

    const pinch = data.pinch;
    if (!pinch) {
      distEl.textContent = "-";
      angEl.textContent = "-";
      progTextEl.textContent = "-";
      progBarEl.style.width = "0%";
      pstateEl.textContent = "UNCALIBRATED";
      return;
    }

    distEl.textContent = `${pinch.distance.toFixed(4)} m`;
    angEl.textContent = `${pinch.angle_deg.toFixed(1)}°`;

    if (pinch.progress == null || !pinch.calibrated) {
      progTextEl.textContent = "-";
      progBarEl.style.width = "0%";
      pstateEl.textContent = "UNCALIBRATED";
    } else {
      const p = Math.max(0, Math.min(1, pinch.progress));
      progTextEl.textContent = p.toFixed(2);
      progBarEl.style.width = `${p * 100}%`;
      pstateEl.textContent = pinch.state;
    }

    if (typeof pinch.threshold === "number") {
      const t = pinch.threshold;
      threshSlider.value = t.toFixed(2);
      threshValueEl.textContent = t.toFixed(2);
    }
  };
}

function sendCommand(cmd) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify(cmd));
}

function sendThreshold() {
  const value = parseFloat(threshSlider.value);
  threshValueEl.textContent = value.toFixed(2);
  sendCommand({
    type: "set_threshold",
    value: value,
  });
}

threshSlider.addEventListener("input", () => {
  sendThreshold();
});

btnOpen.addEventListener("click", () => {
  sendCommand({ type: "set_open" });
});

btnClosed.addEventListener("click", () => {
  sendCommand({ type: "set_closed" });
});

setStatus(false);
connectWS();
