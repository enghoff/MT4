# Grounding DINO server — setup

Open-vocabulary detection for the MT4 desk camera. The detector is a small
FastAPI service wrapping `IDEA-Research/grounding-dino-base`; the arm side
talks to it over HTTP and needs nothing else. Everything else in this repo
works with the service absent — cube pick/place, calibration and the rest of
the MCP tools do not depend on it.

**What it buys you:** an `obj_N` entity for anything the HSV cube detector
cannot name — a pen, a key, a screwdriver — from a text phrase instead of a
hand-read pixel.

---

## Where to run it

The service is just an HTTP endpoint, so put it wherever your GPU is. Pick the
row that matches your hardware:

| Your setup | Where the service runs | Client URL |
|---|---|---|
| **GPU in the machine driving the arm** | Same machine, on loopback | `http://127.0.0.1:8765` (the default — nothing to configure) |
| **GPU on another host** | That host, reached by SSH tunnel or a LAN bind | Tunnel: the default. LAN bind: `http://<gpu-host>:8765` |
| **No GPU** | Same machine, on CPU | The default. Works, but expect seconds per frame |

```
                                       ┌─────────────────────────────┐
 mt4_vision.grounding                  │ server.py (uvicorn)         │
 mt4_mcp  mt4_locate_by_prompt   HTTP   │   GET  /health              │
   MT4_GROUNDING_URL ──────────────────>│   POST /detect              │
                                       │   grounding-dino-base       │
 same host, or across the network       └─────────────────────────────┘
```

The only thing the arm side needs to know is `MT4_GROUNDING_URL` (default
`http://127.0.0.1:8765`). Everything below is about standing the service up.

---

## Install

Needs Python 3.10+ and, for useful speed, a CUDA GPU. The base model was
developed against an 8 GB card with room to spare; if memory is tight,
`IDEA-Research/grounding-dino-tiny` is a drop-in via `GROUNDING_DINO_MODEL` —
faster and less accurate. With no GPU the service still runs, on CPU.

Copy `server.py`, `requirements.txt` and (on Linux) `grounding-dino.service`
from [services/grounding_dino/](../services/grounding_dino/) to wherever you
want it to live, then:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # several GB of wheels, mostly torch

# check the GPU is visible before going further
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# first run downloads ~700 MB of weights into the HuggingFace cache
.venv/bin/python server.py                     # Ctrl-C once it logs "Uvicorn running"
```

Do that first weight download in the foreground, before putting the service
under a supervisor. Restart-on-failure supervisors turn a slow or failing
download into what looks like a crash-loop. Once the weights are cached the
service starts offline.

`requirements.txt` is deliberately unpinned. `transformers` must be ≥ 4.40 for
the Grounding DINO processor; `torchvision` and `accelerate` are listed because
transformers may reach for them internally, though `server.py` imports neither
directly.

### Keeping it running

On a Linux or WSL2 host with systemd, use the bundled unit:

```bash
install -m 644 grounding-dino.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now grounding-dino.service
systemctl status grounding-dino.service
curl http://127.0.0.1:8765/health
```

Edit `WorkingDirectory` and `ExecStart` in the unit if you did not install to
`/opt/grounding_dino`. On Windows or macOS there is no unit file — run
`server.py` under whatever supervisor you already use (NSSM, a Scheduled Task,
`launchd`, a terminal you leave open).

A healthy `/health` reports `ok=True`, `loaded=True`, and `device=cuda` if a GPU
was found.

### Configuration

All environment variables; there is no config file:

| Variable | Default | Notes |
|----------|---------|-------|
| `GROUNDING_DINO_HOST` | `127.0.0.1` | Loopback. See [Reaching another host](#reaching-a-service-on-another-host) before changing it |
| `GROUNDING_DINO_PORT` | `8765` | Must match whatever the client's `MT4_GROUNDING_URL` points at |
| `GROUNDING_DINO_MODEL` | `IDEA-Research/grounding-dino-base` | `…-tiny` is the lighter option |
| `GROUNDING_DINO_DEVICE` | `cuda` if available, else `cpu` | The bundled unit pins `cuda` |

The model loads once at startup (FastAPI `lifespan`), so the first request after
a restart is fast, but the restart itself costs a few seconds of VRAM
allocation.

---

## Reaching a service on another host

Skip this section entirely if the service runs on the machine driving the arm.

**Option A — SSH tunnel.** The service stays on loopback and the tunnel is your
access control. Nothing is exposed to the network.

```bash
ssh -N -L 8765:127.0.0.1:8765 user@gpu-host
```

This repo ships a helper for it,
[scripts/start_grounding_tunnel.ps1](../scripts/start_grounding_tunnel.ps1),
which checks whether the port is already listening (printing `/health` if so),
verifies the key exists, and adds `ExitOnForwardFailure` plus a keepalive.
Its defaults match the reference deployment below, so override them for yours:
`-RemoteHost`, `-User`, `-IdentityFile`, `-LocalPort`, `-RemoteBind`.

**Option B — bind to the LAN.** Set `GROUNDING_DINO_HOST=0.0.0.0` and point
`MT4_GROUNDING_URL` at `http://<gpu-host>:8765`. Simpler, but **the service has
no authentication of any kind** — anyone who can reach the port can post images
to your GPU. Only do this on a network you trust, and prefer Option A otherwise.

### If your GPU host runs WSL2

Two host-level settings matter, and neither is obvious when the service
mysteriously fails to start or becomes unreachable:

- **systemd must be enabled** — `/etc/wsl.conf` inside the distro:

  ```ini
  [boot]
  systemd=true
  ```

  Without it there is no `systemctl` and the unit file is inert.

- **Networking mode decides how you reach it** — `%UserProfile%\.wslconfig` on
  the Windows side:

  ```ini
  [wsl2]
  networkingMode=mirrored
  vmIdleTimeout=-1
  ```

  `mirrored` puts WSL directly on the LAN, so an `sshd` inside WSL is reachable
  at the host's own address with no port-forwarding rules. Without it, WSL sits
  behind NAT: you need a `netsh interface portproxy` rule for port 22, and WSL's
  address changes on every boot. `vmIdleTimeout=-1` stops the VM shutting down
  while idle and taking the service with it. `.wslconfig` changes need
  `wsl --shutdown` to apply.

CUDA needs no work inside WSL beyond a current NVIDIA driver on the Windows
host — the driver supplies the WSL CUDA stubs, and pip's CUDA torch wheels
bring the rest.

### Re-deploying after editing server.py

Push the files, reinstall the unit if it changed, restart, confirm:

```bash
scp server.py requirements.txt grounding-dino.service user@gpu-host:/opt/grounding_dino/
ssh user@gpu-host "install -m 644 /opt/grounding_dino/grounding-dino.service \
  /etc/systemd/system/ && systemctl daemon-reload \
  && systemctl restart grounding-dino.service \
  && sleep 8 && curl -s http://127.0.0.1:8765/health"
```

`scp` alone does not update `/etc/systemd/system/`, so a restart without the
`install` + `daemon-reload` keeps running the old unit. An SSH tunnel survives a
service restart; only in-flight requests fail.

---

## HTTP API

Two endpoints, both in [services/grounding_dino/server.py](../services/grounding_dino/server.py).

### `GET /health`

```json
{"ok": true, "model": "IDEA-Research/grounding-dino-base",
 "device": "cuda", "cuda": true, "loaded": true}
```

### `POST /detect` — `multipart/form-data`

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `image` | file | — | JPEG/PNG frame |
| `prompt` | str | — | Period-separated phrases, e.g. `pen. red cube.` |
| `box_threshold` | float | 0.35 | Box score floor |
| `text_threshold` | float | 0.25 | Phrase-match score floor |

The server lowercases the prompt and appends a trailing `.` if missing — both
are Grounding DINO input requirements, not cosmetics. Response:

```json
{"ok": true, "prompt": "pen.", "width": 1280, "height": 720,
 "detections": [{"label": "pen", "score": 0.62, "box": [x1, y1, x2, y2]}]}
```

Boxes are pixel `xyxy` in the submitted frame. Errors are HTTP status codes:
`400` empty image / undecodable / empty prompt, `503` model not loaded.

The post-processing path is version-tolerant on purpose: it prefers
`post_process_grounded_object_detection` (with and without the positional
`input_ids`, which moved between transformers releases) and falls back to
`post_process_object_detection`, labelling everything with the raw prompt. If a
transformers version ever lands on that last branch, per-phrase labels degrade
to the whole prompt string while boxes stay correct.

---

## Consumers in this repo

| Entry point | What it does |
|-------------|--------------|
| [mt4_vision/grounding.py](../mt4_vision/grounding.py) | Client. `health()`, `detect(frame, prompt)` → `Detection` list sorted by score, with `cx`/`cy`. Raises `GroundingError` when unreachable |
| `python -m mt4_vision grounding --prompt "pen"` | Detect and save `grounding_frame.jpg`. `--locate` measures the top hit, `--pick` picks it, `--label`, `--box-threshold`, `--text-threshold`, `--window` |
| MCP `mt4_locate_by_prompt` | Registers the top hit as `obj_N` with entity, detection list and a view token |
| [move_object_to_marker.py](../move_object_to_marker.py) | Interactive mover: type a description, DINO finds it, it lands on a free marker. Background detection thread refreshes the preview |

Detection stops at a box; measurement is separate. Both the CLI and the MCP tool
hand the box to `locate.measure_with_box_fallback`, which tries three things in
order — **GrabCut inside the DINO box**, then desk-segment `measure` around the
box centre, then the raw box AABB. Only the first two recover a true centre,
long axis and mm size; the AABB fallback inherits whatever slop the box has. So
a plausible box is not yet a pickable object, and `grasp_feasibility` is what
decides.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `grounding service unreachable at http://127.0.0.1:8765` | Service not running, or (remote setups) no tunnel | Check the service; start the tunnel |
| Port already listening but `/health` fails | Stale tunnel, or the service died behind a live one | Check the service on the GPU host; kill the old `ssh` and restart |
| `503 model not loaded` | Requested during startup | Wait a few seconds; `/health` reports `loaded` |
| `/health` shows `device=cpu` unexpectedly | No CUDA torch, or the driver is not visible | `torch.cuda.is_available()` in the venv; check the GPU driver |
| OOM loading the model | Card too small, or shared with other work | Check what else is on the GPU; try `…-tiny` |
| `no detections for prompt='…'` | Threshold too high, or phrasing | Lower `--box-threshold` toward 0.2, and describe appearance rather than function ("thin metal rod" over "stylus") |
| Boxes right, picks miss | A measurement problem, not detection | See the fallback chain above and [docs/CALIBRATION.md](CALIBRATION.md) |
| **Remote:** host unreachable, tunnel drops | Host asleep, or a WSL2 VM shut down | `vmIdleTimeout=-1` prevents the WSL case; otherwise wake the host |
| **Remote:** `ssh` hangs where `ping` also fails | Name resolving to an address that does not route (e.g. a link-local IPv6 from mDNS) | Force IPv4 with `ssh -4`, or use the address directly |
| **Remote:** tunnel exits immediately | Something else owns the local port | Free it, or pass a different local port |

---

## Reference deployment

Not requirements — just the setup this was developed and verified against, for
comparison if something behaves differently for you.

| | |
|---|---|
| GPU host | a LAN machine (`media`) running WSL2 under Windows, reached by SSH |
| GPU | NVIDIA GeForce RTX 3070, 8 GB |
| Service dir | `/opt/grounding_dino` (`server.py`, `requirements.txt`, the unit, `.venv/`) |
| Python | 3.10.12 |
| Resolved packages | torch 2.11.0+cu128, torchvision 0.26.0+cu128, transformers 5.14.1, fastapi 0.141.1, uvicorn 0.52.0 |
| Weight cache | `/root/.cache/huggingface` — 1.6 GB holding both `-base` and `-tiny` |
| Bind | `127.0.0.1:8765`, loopback only, reached by SSH local forward |
| Access | key-only SSH, `PasswordAuthentication no` |

In that deployment the GPU host resolves by mDNS to both a LAN IPv4 address and
a link-local IPv6 one. The v6 record does not route, so a bare `ping` or `ssh`
picks it and hangs — which is why the `ssh`/`scp` invocations for it pass `-4`.
Worth knowing generally: if your GPU host resolves via mDNS, expect the same.

---

## See also

- [services/grounding_dino/README.md](../services/grounding_dino/README.md) — what each deployed file is, plus the day-to-day detect commands
- [README.md § Open-vocabulary objects](../README.md#open-vocabulary-objects) — how `obj_N` entities fit the wider stack
- [docs/CALIBRATION.md](CALIBRATION.md) — the measurement layers a DINO box feeds into
