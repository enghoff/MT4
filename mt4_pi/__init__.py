"""pi0.5 (OpenPI) remote-inference integration for the MT4.

Talks to the pi0.5 policy server (https://github.com/enghoff/pi0.5-server)
over WebSocket and turns action chunks into MT4 motion via the existing
mt4_jog kinematics and firmware `mq` queue.

Execution is inert until an MT4-specific fine-tune exists -- see runtime.py
and pi0.5-server's docs/mt4-client-integration.md. The server currently
serves pi05_droid, a 7-DoF Franka policy that has never seen an MT4; its
action chunks are dimensionally valid and finite but are not real MT4
commands.
"""
