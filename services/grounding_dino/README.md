# Grounding DINO service

Open-vocabulary detector for the MT4 desk camera. A FastAPI service you run
wherever your GPU is — the same machine that drives the arm, or another host
reached over an SSH tunnel or a LAN bind. It also runs on CPU, slowly.

**Install, supervision, remote access, HTTP API and troubleshooting:
[docs/GROUNDING_DINO.md](../../docs/GROUNDING_DINO.md).** Deploy steps live
only there, so there is one copy to keep correct.

These three files are what you deploy:

| File | |
|------|--|
| `server.py` | FastAPI service — `GET /health`, `POST /detect` |
| `grounding-dino.service` | systemd unit for Linux/WSL2 hosts; edit the paths if you install elsewhere |
| `requirements.txt` | venv deps for a from-scratch install |

Day-to-day, from the machine driving the arm:

```powershell
python -m mt4_vision --camera 1 grounding --prompt "pen"           # detect
python -m mt4_vision --camera 1 grounding --prompt "pen" --locate  # + measure
```

Add `.\scripts\start_grounding_tunnel.ps1` first, left running, if the service
is on another host and you reach it by SSH forward.

Env: `MT4_GROUNDING_URL` (default `http://127.0.0.1:8765`).
Model: `IDEA-Research/grounding-dino-base` (`GROUNDING_DINO_MODEL`).

MCP: `mt4_locate_by_prompt`.
