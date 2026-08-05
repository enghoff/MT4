# SSH local forwards to the GPU services running on another host.
#
# Only needed when the services are remote and bound to that host's loopback.
# This opens the same ports here, so a client reaches each one at
# http://127.0.0.1:<port> exactly as if it ran on this machine. If a service
# runs on this machine, you do not need this for it.
#
# One ssh connection carries every forward, so this is one window, not one per
# service. A forward costs nothing when the far end is stopped -- which is the
# normal state of one of grounding-dino/qwen3-vl, since they share a card and
# swap (services/switch_vision_service.sh) -- so the default forwards both and
# you do not restart the tunnel when you switch.
#
# SAM 2.1 runs in-process on the arm host (mt4_vision.sam); it is not tunneled.
# Override -RemoteHost / -User / -IdentityFile for your GPU host.
#
# Usage:
#   .\scripts\start_tunnel.ps1                     # dino + qwen
#   .\scripts\start_tunnel.ps1 qwen                # just qwen
#   .\scripts\start_tunnel.ps1 -LocalPorts @{qwen=18766}
#   # leave running, then from another terminal:
#   curl http://127.0.0.1:8766/health

param(
    # Names, or "all". A PowerShell prompt parses `qwen,dino` into two arguments
    # on its own; `powershell -File … -Service qwen,dino` hands it over as one
    # string, so the split below covers both rather than failing on the second.
    [string[]]$Service = @("all"),
    [string]$RemoteHost = "media",
    [string]$User = "root",
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\id_ed25519_media",
    # Service -> local port, when the default is taken by something else here.
    # The remote side is always the service's own port; only this end moves, so
    # point the client's URL variable at it too.
    [hashtable]$LocalPorts = @{},
    [int]$RetrySeconds = 5
)

$ErrorActionPreference = "Stop"

# Port, systemd unit and the environment variable a client reads, per service.
# The ports are the service defaults; changing one means changing the unit on
# the GPU host and the client variable together.
$Known = [ordered]@{
    dino = @{ Port = 8765; Unit = "grounding-dino.service"; Env = "MT4_GROUNDING_URL" }
    qwen = @{ Port = 8766; Unit = "qwen3-vl.service";       Env = "MT4_QWEN_URL" }
}

$asked = @()
foreach ($item in $Service) {
    foreach ($part in ($item -split ",")) {
        if ($part.Trim()) { $asked += $part.Trim().ToLower() }
    }
}
if ($asked -contains "all") {
    $wanted = @($Known.Keys)
} else {
    $wanted = @($asked | Select-Object -Unique)
}

foreach ($name in ($wanted + @($LocalPorts.Keys))) {
    if (-not $Known.Contains($name)) {
        Write-Error "unknown service '$name'; known: $($Known.Keys -join ', '), all"
    }
}

if (-not (Test-Path $IdentityFile)) {
    Write-Error "SSH key not found: $IdentityFile"
}

# Forward only what is not already listening. ExitOnForwardFailure is what
# makes a dead tunnel visible rather than silently half-open, and it also means
# ssh refuses the whole connection when ANY requested local port is taken -- so
# one stale forward from an earlier window would otherwise take the rest down
# with it.
$forwards = @()
$skipped = @()
foreach ($name in $wanted) {
    $port = $Known[$name].Port
    if ($LocalPorts.Contains($name)) { $port = [int]$LocalPorts[$name] }

    $busy = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($busy) {
        $health = "not answering /health"
        try {
            $reply = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3
            $health = ($reply | ConvertTo-Json -Compress)
        } catch {
            # Kept as the default text: a listening port that will not answer is
            # a stale tunnel or a dead service behind a live one, and saying so
            # beats claiming the service is up.
        }
        Write-Host "$name`: port $port already listening -- $health"
        $skipped += $name
        continue
    }
    $forwards += "-L"
    $forwards += "${port}:127.0.0.1:$($Known[$name].Port)"
    Write-Host "$name`: localhost:$port -> ${User}@${RemoteHost}:127.0.0.1:$($Known[$name].Port)"
}

if ($forwards.Count -eq 0) {
    Write-Host ""
    Write-Host "Nothing left to forward ($($skipped -join ', ') already up). Exiting."
    exit 0
}

Write-Host ""
Write-Host "Reconnects automatically. Leave this window open. Ctrl+C to stop."
Write-Host ""

# A single ssh invocation dies with the first network hiccup ("client_loop:
# send disconnect: Connection reset"), and nothing says so except every
# subsequent request failing as "service unreachable". That is a bad failure
# mode for ask_qwen.py, which runs a multi-step instruction to completion with
# the arm mid-task -- so reconnect, and timestamp each drop so the log shows
# how flaky the link really is.
$attempt = 0
while ($true) {
    $attempt++
    $since = Get-Date -Format "HH:mm:ss"
    if ($attempt -gt 1) { Write-Host "[$since] reconnecting (attempt $attempt)..." }

    $sshArgs = @(
        "-4", "-N",
        "-i", $IdentityFile,
        "-o", "IdentitiesOnly=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "StrictHostKeyChecking=accept-new"
    ) + $forwards + @("${User}@${RemoteHost}")
    & ssh @sshArgs

    $code = $LASTEXITCODE
    $now = Get-Date -Format "HH:mm:ss"
    Write-Host "[$now] tunnel exited (code $code) after starting at $since"

    # Every port we opened is listening again without us: another window won
    # the race, and retrying would spin forever against forwards we do not own.
    $ours = @()
    for ($i = 0; $i -lt $forwards.Count; $i += 2) {
        $ours += [int]($forwards[$i + 1] -split ":")[0]
    }
    $taken = @($ours | Where-Object {
        Get-NetTCPConnection -LocalPort $_ -State Listen -ErrorAction SilentlyContinue
    })
    if ($taken.Count -eq $ours.Count) {
        Write-Host "Ports $($ours -join ', ') are listening again from elsewhere - stopping."
        break
    }
    Start-Sleep -Seconds $RetrySeconds
}
