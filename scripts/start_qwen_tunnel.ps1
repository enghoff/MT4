# SSH local forward to the Qwen3-VL service running on another host.
#
# Only needed when the service is remote and bound to that host's loopback.
# This opens the same port here so an HTTP client can reach it at
# http://127.0.0.1:8766. If the service runs on this machine, you do not
# need this.
#
# The defaults below match the reference deployment (see docs/QWEN3-VL.md) --
# override -RemoteHost / -User / -IdentityFile for yours.
#
# Usage:
#   .\scripts\start_qwen_tunnel.ps1
#   # leave running, then from another terminal:
#   curl http://127.0.0.1:8766/health

param(
    [int]$LocalPort = 8766,
    [string]$RemoteHost = "media",
    [string]$RemoteBind = "127.0.0.1:8766",
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
