# SSH local forward to the SAM 2.1 segmentation service running on another host.
#
# Only needed when the service is remote and bound to that host's loopback.
# This opens the same port here so an HTTP client can reach it at
# http://127.0.0.1:8767. If the service runs on this machine, you do not
# need this.
#
# The defaults below match the reference deployment (see docs/SAM2.md) --
# override -RemoteHost / -User / -IdentityFile for yours.
#
# Usage:
#   .\scripts\start_sam_tunnel.ps1
#   # leave running, then from another terminal:
#   curl http://127.0.0.1:8767/health

param(
    [int]$LocalPort = 8767,
    [string]$RemoteHost = "media",
    [string]$RemoteBind = "127.0.0.1:8767",
    [string]$User = "root",
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\id_ed25519_media",
    [int]$RetrySeconds = 5
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
Write-Host "Reconnects automatically. Leave this window open. Ctrl+C to stop."
Write-Host ""

# A single ssh invocation dies with the first network hiccup ("client_loop:
# send disconnect: Connection reset"), and nothing says so except every
# subsequent request failing as "service unreachable" -- so reconnect, and
# timestamp each drop so the log shows how flaky the link really is.
$attempt = 0
while ($true) {
    $attempt++
    $since = Get-Date -Format "HH:mm:ss"
    if ($attempt -gt 1) { Write-Host "[$since] reconnecting (attempt $attempt)..." }

    ssh -4 -N `
        -i $IdentityFile `
        -o IdentitiesOnly=yes `
        -o ExitOnForwardFailure=yes `
        -o ServerAliveInterval=30 `
        -o ServerAliveCountMax=3 `
        -o StrictHostKeyChecking=accept-new `
        -L "${LocalPort}:${RemoteBind}" `
        "${User}@${RemoteHost}"

    $code = $LASTEXITCODE
    $now = Get-Date -Format "HH:mm:ss"
    Write-Host "[$now] tunnel exited (code $code) after starting at $since"

    # A local port already in use means another tunnel won the race; retrying
    # would spin forever against a working forward we did not open.
    if (Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue) {
        Write-Host "Port $LocalPort is listening again from elsewhere - stopping."
        break
    }
    Start-Sleep -Seconds $RetrySeconds
}
