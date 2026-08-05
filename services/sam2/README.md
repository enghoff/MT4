# SAM 2.1 service

Prompted segmentation for the MT4 desk camera: a point or a box in, a binary
mask out. A FastAPI service you run wherever your GPU is — the same machine
that drives the arm, or another host reached over an SSH tunnel or a LAN bind.
It also runs on CPU, slowly.

**Install, supervision, remote access, HTTP API, the measured optimizations and
troubleshooting: [docs/SAM2.md](../../docs/SAM2.md).** Deploy steps live only
there, so there is one copy to keep correct.

These three files are what you deploy:

| File | |
|------|--|
| `server.py` | FastAPI service — `GET /health`, `POST /embed`, `POST /segment` |
| `sam2.service` | systemd unit for Linux/WSL2 hosts; edit the paths if you install elsewhere |
| `requirements.txt` | venv deps for a from-scratch install |

Day-to-day, from the machine driving the arm:

```powershell
python -m mt4_vision --camera 1 sam --pixel 737 570            # mask at a pixel
python -m mt4_vision --camera 1 sam --box 671 523 787 647      # mask in a box
python -m mt4_vision --camera 1 sam --pixel 737 570 --candidates
```

Add `.\scripts\start_tunnel.ps1` first, left running, if the service is on
another host and you reach it by SSH forward.

Env: `MT4_SAM_URL` (default `http://127.0.0.1:8767`).
Model: `facebook/sam2.1-hiera-small` in fp16 (`SAM2_MODEL`, `SAM2_DTYPE`).

The card it shares with `qwen3-vl.service` has room for both: this one holds
~600 MiB including its CUDA context.
