# Grounding DINO service (media)

Open-vocabulary detector for the MT4 desk camera. Runs on the GPU host
`media` (RTX 3070) and is reached from the MT4 Windows machine through an
SSH local forward.

**Setup, re-deploy, HTTP API and troubleshooting:
[docs/GROUNDING_DINO.md](../../docs/GROUNDING_DINO.md).** Deploy steps live
only there, so there is one copy to keep correct.

Files here are the source of truth for `/opt/grounding_dino` on media:

| File | |
|------|--|
| `server.py` | FastAPI service — `GET /health`, `POST /detect` |
| `grounding-dino.service` | systemd unit; also installed to `/etc/systemd/system/` |
| `requirements.txt` | venv deps for a from-scratch install |

Day-to-day, from the MT4 host:

```powershell
.\scripts\start_grounding_tunnel.ps1                            # leave running
python -m mt4_vision --camera 1 grounding --prompt "pen"           # detect
python -m mt4_vision --camera 1 grounding --prompt "pen" --locate  # + measure
```

Env: `MT4_GROUNDING_URL` (default `http://127.0.0.1:8765`).
Model on media: `IDEA-Research/grounding-dino-base` (`GROUNDING_DINO_MODEL`).

MCP: `mt4_locate_by_prompt`.
