# SSH local forward to a Grounding DINO service running on another host.
#
# Only needed when the service is remote and bound to that host's loopback.
# This opens the same port here so mt4_vision.grounding (MT4_GROUNDING_URL)
# can reach it. If the service runs on this machine, you do not need this.
#
# The defaults below match the setup this was developed against -- override
# -RemoteHost / -User / -IdentityFile for yours. See docs/GROUNDING_DINO.md.
#
# Usage:
#   .\scripts\start_grounding_tunnel.ps1
#   # leave running, then from another terminal:
#   python -m mt4_vision grounding --prompt "pen"

param(
    [int]$LocalPort = 8765,
    [string]$RemoteHost = "media",
    [string]$RemoteBind = "127.0.0.1:8765",
    [string]$User = "root",
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\id_ed25519_media"
)

$ErrorActionPreference = "Stop"

$existing = Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Port $LocalPort already listening (tunnel or service already up)."
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$LocalPort/health" -TimeoutSec 3
        Write-Host ("health: " + ($health | ConvertTo-Json -Compress))
    } catch {
        Write-Warning "Port is open but /health failed: $_"
    }
    exit 0
}

if (-not (Test-Path $IdentityFile)) {
    Write-Error "SSH key not found: $IdentityFile"
}

Write-Host "Tunnel: localhost:$LocalPort -> ${User}@${RemoteHost}:$RemoteBind"
Write-Host "Leave this window open. Ctrl+C to stop."
Write-Host ""

ssh -4 -N `
    -i $IdentityFile `
    -o IdentitiesOnly=yes `
    -o ExitOnForwardFailure=yes `
    -o ServerAliveInterval=30 `
    -L "${LocalPort}:${RemoteBind}" `
    "${User}@${RemoteHost}"
