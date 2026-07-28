# pi0.5 Model Deployment — MEDIA Inference Server

How to check, start, stop, and verify the pi0.5 policy server that MT4's
`mt4_pi` package talks to for remote inference. This is the client-repo view;
the server-side deploy tooling lives in a separate repo (below).

---

## 1. Topology

| Piece | Where |
|---|---|
| Inference server (`serve_policy.py`, OpenPI) | `MEDIA` (`192.168.1.3`), inside a WSL2 distro, RTX 3070 (8 GB) |
| Server repo (`pi0.5-server`) | Checked out on `MEDIA` at `/mnt/z/pi0.5` (a network share mounted *inside MEDIA's WSL* — it is not reachable as `Z:` from other machines) |
| Client (this repo) | `mt4_pi/policy_client.py` wraps `openpi_client.WebsocketClientPolicy`; `mt4_pi/adapter.py` converts the returned action chunks into MT4 waypoints — see the wire-contract link in its docstring |
| Wire protocol | WebSocket, msgpack, port `8000` |

Access is root SSH, key-based (`~/.ssh/id_ed25519`, comment `enghoff`) — `ssh
root@192.168.1.3` needs no `-i` flag since that's the default identity file.

---

## 2. Check current state

```bash
# Is a server answering?
curl http://192.168.1.3:8000/healthz          # -> OK, or connection refused

# GPU memory in use
ssh root@192.168.1.3 'nvidia-smi --query-gpu=memory.total,memory.used --format=csv'
```

Baseline idle is ~600 MiB used. With `pi05_droid` loaded it's ~7.9 GiB —
**the 3070's 8 GB is nearly maxed by that one checkpoint alone**, so there's
no headroom to run two models at once, and a meaningfully larger fine-tune
may not fit at all.

---

## 3. Start / redeploy a checkpoint

Deploying is entirely the server repo's job — `deploy_checkpoint.sh` stops
whatever's running, starts the new checkpoint fully detached (survives the
SSH session ending), waits for `/healthz`, then fires a synthetic warm-up
call so the ~33 s JAX-compile cost lands on the deploy, not the first real
client:

```bash
ssh root@192.168.1.3 'bash /mnt/z/pi0.5/deploy_checkpoint.sh \
    <checkpoint-dir> [config] [mode] [port]'
```

To bring back the stock pretrained checkpoint:

```bash
ssh root@192.168.1.3 'bash /mnt/z/pi0.5/deploy_checkpoint.sh \
    /root/.cache/openpi/openpi-assets/checkpoints/pi05_droid pi05_droid gpu 8000'
```

For an actual fine-tune, copy it onto `MEDIA` first — `deploy_checkpoint.sh`
does not fetch it for you:

```bash
rsync -avz ./my_finetuned_ckpt/ \
    root@192.168.1.3:/root/.cache/openpi/openpi-assets/checkpoints/my_finetuned_ckpt/
```

(destination just needs a `params/` subdirectory). Full detail on config
names, rollback, and troubleshooting is in the server repo at
`/mnt/z/pi0.5/docs/remote-deploy.md` (readable from an SSH session on
`MEDIA`; not reachable from this machine directly since the share isn't
mounted here).

---

## 4. Stop the server

Prefer killing the exact PID — `deploy_checkpoint.sh` prints it
(`server pid: N`) on deploy:

```bash
ssh root@192.168.1.3 'kill <pid>'
```

**Gotcha:** don't run `pkill -f serve_policy` as part of a multi-line/heredoc
SSH command. `pkill -f` matches against full command lines, and the *shell
invoking your script* has the literal text `serve_policy` in its own argv
(because it's part of the script string sshd handed it) — so it self-matches
and kills your SSH session instead of cleanly stopping the server. Hit this
2026-07-28. If you don't have the PID handy, use the bracket trick to dodge
self-matching:

```bash
ssh root@192.168.1.3 "pkill -f '[s]erve_policy'"
```

`[s]erve_policy` matches the target process's real command line the same as
`serve_policy` would, but as a literal string it doesn't appear in your own
invocation, so it can't match itself.

---

## 5. Verify from the client side

No camera or arm needed — this sends a synthetic DROID-shaped observation
and checks the response:

```bash
python -m mt4_pi.policy_client --host 192.168.1.3 --port 8000 --iters 3
```

Expect `(15, 8)` action chunks, `finite=True`, and after the first call
(compile-warmed by the deploy script already) latency in the ~200–500 ms
range. Anything wildly higher, or `finite=False`, points at a bad checkpoint
or config mismatch rather than a network problem.

---

## See also

- `mt4_pi/adapter.py` — the DROID action → MT4 waypoint mapping (which
  columns are used, units, the velocity-vs-position gripper distinction)
- `mt4_pi/policy_client.py` — connection defaults and the keepalive
  (`ping_interval`) workaround for the JAX-compile stall on first call
- pi0.5-server repo, on `MEDIA` at `/mnt/z/pi0.5/docs/`:
  `remote-deploy.md` (this doc's server-side counterpart, more detail),
  `mt4-client-integration.md` (wire contract), `pi05-mt4-remote-inference.md`
