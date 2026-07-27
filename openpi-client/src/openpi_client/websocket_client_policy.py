import logging
import os
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        ping_interval: Optional[float] = -1.0,
    ) -> None:
        # LOCAL PATCH. `infer` blocks in recv() for the whole inference, so if the
        # server takes longer than the websockets keepalive timeout (default 20 s)
        # the client tears the connection down with 1011 "keepalive ping timeout"
        # mid-request. Irrelevant on GPU (~100 ms) but fatal on CPU (~25-30 s).
        #
        # ping_interval: None disables keepalive; a number sets the interval;
        # the -1.0 sentinel means "consult OPENPI_WS_PING_INTERVAL, else use the
        # websockets default", so upstream behaviour is unchanged by default.
        if ping_interval == -1.0:
            raw = os.environ.get("OPENPI_WS_PING_INTERVAL")
            if raw is None:
                ping_interval = -1.0
            else:
                ping_interval = None if raw.strip().lower() in ("none", "0", "") else float(raw)
        self._ping_interval = ping_interval

        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                kwargs = {} if self._ping_interval == -1.0 else {"ping_interval": self._ping_interval}
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers, **kwargs
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
