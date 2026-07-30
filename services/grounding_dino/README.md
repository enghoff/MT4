# Grounding DINO service (media)

Open-vocabulary detector for the MT4 desk camera. Runs on the GPU host
`media` (RTX 3070) and is reached from the MT4 Windows machine through an
SSH local forward.

## On media

```bash
# already deployed to /opt/grounding_dino and enabled as systemd:
systemctl status grounding-dino.service
curl http://127.0.0.1:8765/health
```

Re-deploy from this repo:

```powershell
scp -4 -i $env:USERPROFILE\.ssh\id_ed25519_media -o IdentitiesOnly=yes `
  server.py requirements.txt grounding-dino.service root@media:/opt/grounding_dino/
ssh -4 -i $env:USERPROFILE\.ssh\id_ed25519_media -o IdentitiesOnly=yes root@media `
  "systemctl restart grounding-dino.service"
```

## On the MT4 host

```powershell
# leave running
.\scripts\start_grounding_tunnel.ps1

# detect
python -m mt4_vision --camera 1 grounding --prompt "pen"
# detect + measure
python -m mt4_vision --camera 1 grounding --prompt "pen" --locate
```

Env: `MT4_GROUNDING_URL` (default `http://127.0.0.1:8765`).
Model on media: `IDEA-Research/grounding-dino-base` (`GROUNDING_DINO_MODEL`).

MCP: `mt4_locate_by_prompt`.
