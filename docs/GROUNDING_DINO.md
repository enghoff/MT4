# Grounding DINO server — setup

Open-vocabulary detection for the MT4 desk camera. The MT4 Windows machine has
no CUDA, so the detector runs on a separate GPU host (`media`) as a systemd
service bound to loopback, and the MT4 host reaches it through an SSH local
forward. Nothing else in the repo depends on it — cube pick/place, calibration
and the rest of the MCP tools work with the service down.

**What it buys you:** an `obj_N` entity for anything the HSV cube detector
cannot name — a pen, a key, a screwdriver — from a text phrase instead of a
hand-read pixel.

```
MT4 host (Windows, no GPU)                     media (RTX 3070)
┌──────────────────────────────┐               ┌─────────────────────────────┐
│ mt4_vision.grounding         │               │ grounding-dino.service      │
│   MT4_GROUNDING_URL          │  ssh -L 8765  │   uvicorn 127.0.0.1:8765    │
│   http://127.0.0.1:8765 ─────┼──────────────>│   grounding-dino-base @cuda │
│ mt4_mcp  mt4_locate_by_prompt│               │   /opt/grounding_dino       │
└──────────────────────────────┘               └─────────────────────────────┘
```

---

## As deployed today

Verified live on 2026-07-30.

| | |
|---|---|
| Host | `media` — WSL2 (Ubuntu, kernel 6.18 microsoft-standard-WSL2) on Windows |
| GPU | NVIDIA GeForce RTX 3070, 8 GB |
| Service dir | `/opt/grounding_dino` (`server.py`, `requirements.txt`, `grounding-dino.service`, `.venv/`) |
| Python | 3.10.12 in `/opt/grounding_dino/.venv` |
| Key packages | torch 2.11.0+cu128, torchvision 0.26.0+cu128, transformers 5.14.1, fastapi 0.141.1, uvicorn 0.52.0 |
| Model | `IDEA-Research/grounding-dino-base` |
| Weight cache | `/root/.cache/huggingface` — 1.6 GB (`grounding-dino-base` + `grounding-dino-tiny`) |
| Bind | `127.0.0.1:8765`, loopback only — never LAN-exposed |
| Unit | `grounding-dino.service`, enabled, `Restart=on-failure` |
| Login | `root@media`, key `~/.ssh/id_ed25519_media`, `PasswordAuthentication no` |

Source of truth for the deployed files is [services/grounding_dino/](../services/grounding_dino/)
in this repo; `/opt/grounding_dino` is a copy pushed by `scp` (see
[Re-deploy](#re-deploy-after-editing-serverpy)).

---

## Client-side setup (MT4 host)

This is all that is needed on the robot machine — no CUDA, no model download.

1. **SSH key.** `~/.ssh/id_ed25519_media`, with its public half in
   `root@media:/root/.ssh/authorized_keys`. Password auth is off, so the key is
   the only way in.

2. **Hostname.** `media` resolves by mDNS to `MEDIA.local`. Confirm it:

   ```powershell
   Resolve-DnsName media
   # MEDIA.local  192.168.1.3        A
   # MEDIA.local  fe80::…%10         AAAA
   ```

   The AAAA record is a **link-local** address that does not route — plain
   `ping media` picks it and times out. That is why every `ssh`/`scp` command
   in this repo passes `-4`. Drop the `-4` and the connection hangs.

3. **Start the tunnel** and leave the window open:

   ```powershell
   .\scripts\start_grounding_tunnel.ps1
   ```

   It checks whether 8765 is already listening (and prints `/health` if so),
   verifies the key exists, then runs an `ssh -4 -N -L 8765:127.0.0.1:8765`
   with `ExitOnForwardFailure=yes` and a 30 s keepalive. Parameters if the
   deployment ever moves: `-LocalPort`, `-RemoteHost`, `-RemoteBind`, `-User`,
   `-IdentityFile`.

4. **Verify end to end:**

   ```powershell
   Invoke-RestMethod http://127.0.0.1:8765/health
   python -m mt4_vision --camera 1 grounding --prompt "pen"
   ```

   A healthy response: `ok=True`, `device=cuda`, `cuda=True`, `loaded=True`,
   `model=IDEA-Research/grounding-dino-base`.

`MT4_GROUNDING_URL` overrides the base URL (default `http://127.0.0.1:8765`) if
you point the client somewhere else.

---

## Server-side setup on media

### Prerequisites specific to WSL2

The service lives inside WSL, which adds two requirements that are easy to miss
and both already satisfied on `media`:

- **systemd inside WSL** — `/etc/wsl.conf` must contain:

  ```ini
  [boot]
  systemd=true
  ```

  Without it there is no `systemctl` and the unit file is inert.

- **Reachable sshd and a VM that stays up** — `C:\Users\<user>\.wslconfig` on
  media's Windows side:

  ```ini
  [wsl2]
  memory=22GB
  processors=8
  swap=8GB
  networkingMode=mirrored
  vmIdleTimeout=-1

  [experimental]
  hostAddressLoopback=true
  ```

  `networkingMode=mirrored` puts WSL on the LAN directly (`eth0` is
  `192.168.1.3/24`), so `sshd` listening on `0.0.0.0:22` inside WSL is
  reachable as `media` with no Windows port-proxy rules. `vmIdleTimeout=-1`
  stops the VM from shutting down while idle and taking the service with it.
  Changing `.wslconfig` needs `wsl --shutdown` to take effect.

  On a non-mirrored WSL install you would instead need a `netsh interface
  portproxy` rule for port 22, and WSL's NAT address changes on every boot —
  mirrored mode is what makes this setup stable.

CUDA itself needs no work inside WSL beyond a current NVIDIA driver on the
Windows host; the WSL CUDA stubs come from the driver, and `pip`'s `+cu128`
torch wheels bring the rest.

### From scratch

```bash
# on media, as root
apt-get install -y python3-venv
mkdir -p /opt/grounding_dino
```

Copy the three files from this repo (see the `scp` line below), then:

```bash
cd /opt/grounding_dino
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # ~5 GB of wheels, mostly torch

# sanity-check CUDA before wiring up systemd
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# first run downloads ~700 MB of weights into /root/.cache/huggingface
.venv/bin/python server.py                     # Ctrl-C once it logs "Uvicorn running"

install -m 644 grounding-dino.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now grounding-dino.service
systemctl status grounding-dino.service
curl http://127.0.0.1:8765/health
```

Do the first weight download interactively rather than via systemd — the unit
has `Restart=on-failure`, and a slow or failing download otherwise looks like a
crash-loop. Once cached, the service starts offline.

`requirements.txt` is deliberately unpinned; the versions in
[As deployed today](#as-deployed-today) are what that resolved to. `transformers`
must be ≥ 4.40 for the Grounding DINO processor.

### Service configuration

Everything is environment variables in the unit file — there is no config file:

| Variable | Default | Notes |
|----------|---------|-------|
| `GROUNDING_DINO_HOST` | `127.0.0.1` | Keep it on loopback; the tunnel is the access control |
| `GROUNDING_DINO_PORT` | `8765` | Must match `-RemoteBind` in the tunnel script |
| `GROUNDING_DINO_MODEL` | `IDEA-Research/grounding-dino-base` | `…-tiny` is also cached — faster, less accurate |
| `GROUNDING_DINO_DEVICE` | `cuda` if available, else `cpu` | The unit pins `cuda`; CPU works but is seconds per frame |

The model loads once at startup (FastAPI `lifespan`), so the first request after
a restart is fast but the restart itself takes a few seconds of VRAM allocation.

### Re-deploy after editing server.py

```powershell
cd services\grounding_dino
scp -4 -i $env:USERPROFILE\.ssh\id_ed25519_media -o IdentitiesOnly=yes `
  server.py requirements.txt grounding-dino.service root@media:/opt/grounding_dino/
ssh -4 -i $env:USERPROFILE\.ssh\id_ed25519_media -o IdentitiesOnly=yes root@media `
  "systemctl restart grounding-dino.service; sleep 8; curl -s http://127.0.0.1:8765/health"
```

`scp` does not copy the unit file into `/etc/systemd/system/` — if
`grounding-dino.service` itself changed, follow with `install -m 644
grounding-dino.service /etc/systemd/system/ && systemctl daemon-reload`.
The tunnel survives a service restart; only in-flight requests fail.

> The stray `services/grounding_dino/.venv/` on the MT4 host is a Windows
> Python 3.12 venv left over from local experimentation. It is gitignored, is
> not what runs the service, and can be deleted.

---

## HTTP API

Two endpoints, both defined in [services/grounding_dino/server.py](../services/grounding_dino/server.py).

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
transformers upgrade ever silently lands on that last branch, per-phrase labels
degrade to the whole prompt string while boxes stay correct.

---

## Consumers in this repo

| Entry point | What it does |
|-------------|--------------|
| [mt4_vision/grounding.py](../mt4_vision/grounding.py) | Client. `health()`, `detect(frame, prompt)` → `Detection` list sorted by score, with `cx`/`cy`. Raises `GroundingError` naming the tunnel script when unreachable |
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
| `grounding service unreachable at http://127.0.0.1:8765` | Tunnel not running (most common) | `.\scripts\start_grounding_tunnel.ps1` |
| Tunnel script: `Port 8765 already listening` but `/health` fails | Stale forward, or the service died behind a live tunnel | Check `systemctl status` on media; kill the old `ssh` and restart |
| `ssh media` hangs, `ping media` times out | mDNS returned the link-local IPv6 | Add `-4`; every ssh/scp in this repo already has it |
| `ExitOnForwardFailure` — tunnel exits immediately | Something else owns local 8765 | `netstat -ano \| findstr :8765`, then free it or pass `-LocalPort` |
| `503 model not loaded` | Requested during startup | Wait a few seconds; `/health` reports `loaded` |
| `/health` shows `device=cpu` | Driver or CUDA broke inside WSL | `nvidia-smi` in WSL, restart the service; check the Windows NVIDIA driver |
| Whole host unreachable, tunnel drops | WSL VM shut down | `vmIdleTimeout=-1` in `.wslconfig` should prevent this; otherwise `wsl -d <distro> -- true` from Windows to restart it |
| `no detections for prompt='…'` | Threshold too high, or phrasing | Lower `--box-threshold` toward 0.2, and describe appearance rather than function ("thin metal rod" over "stylus") |
| Boxes right, picks miss | A measurement problem, not detection | See the fallback chain above and [docs/CALIBRATION.md](CALIBRATION.md) |

Two service-side notes worth remembering: the RTX 3070 has 8 GB and other work
on `media` shares it, so an OOM at startup is a co-tenancy problem rather than a
model-size problem — check `nvidia-smi` before shrinking anything. And `media`
has been knocked off the network before by heavy GPU jobs; if the host stops
answering during unrelated training work, that is the known failure, not the
DINO service.

---

## See also

- [services/grounding_dino/README.md](../services/grounding_dino/README.md) — what each deployed file is, plus the day-to-day detect commands
- [README.md § Open-vocabulary objects](../README.md#open-vocabulary-objects) — how `obj_N` entities fit the wider stack
- [docs/CALIBRATION.md](CALIBRATION.md) — the measurement layers a DINO box feeds into
