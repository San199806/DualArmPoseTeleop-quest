"""In-process stand-in for the Quest FullBodyTarget websocket.

Quest posts the same JSON that Tools/mock_daemon_8000.py used to store, on
ws://<host>:8000/api/move/ws/set_target. This server keeps the latest target
in memory so teleop does not need a second process.
"""
import base64
import hashlib
import json
import socket
import struct
import sys
import threading


_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_TARGET_PATH = "/api/move/ws/set_target"
_STATE_PATH = "/api/state/full_state"
_MAX_FRAME = 1_000_000


def accept_key(key):
    digest = hashlib.sha1((key+_WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


class QuestTargetServer:
    """Accept FullBodyTarget JSON and remember the latest object."""

    def __init__(self, on_payload, host="0.0.0.0", port=8000):
        self._on_payload = on_payload
        self.host = host
        self.port = int(port)
        self._stop = threading.Event()
        self._sock = None
        self._thread = None
        self._clients = []
        self._clients_lock = threading.Lock()

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host, self.port))
        except OSError as error:
            sock.close()
            raise RuntimeError(
                f"Cannot listen for Quest on {self.host}:{self.port}: {error}"
            ) from error
        sock.listen(4)
        sock.settimeout(0.2)
        # Pose reads have to preempt the IK loop, whose C calls hold the GIL for a whole solve.
        sys.setswitchinterval(0.001)
        self.port = sock.getsockname()[1]
        self._sock = sock
        self._thread = threading.Thread(target=self._accept_loop, name="quest-target", daemon=True)
        self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        with self._clients_lock:
            clients = list(self._clients)
        for client in clients:
            try:
                client.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                client, _address = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self._clients_lock:
                self._clients.append(client)
            threading.Thread(
                target=self._handle, args=(client,), name="quest-client", daemon=True,
            ).start()

    def _handle(self, client):
        ack_stop = threading.Event()
        acker = threading.Thread(
            target=_keep_quickack, args=(client, ack_stop), name="quest-quickack", daemon=True,
        )
        try:
            _prepare_tcp(client)
            acker.start()
            client.settimeout(0.2)
            header = _read_headers(client)
            if header is None:
                return
            path = header["path"].split("?", 1)[0]
            if _is_websocket(header) and path == _TARGET_PATH:
                _handshake(client, header)
                self._socket_loop(client)
                return
            if header["method"] == "GET" and path == _STATE_PATH:
                _http_json(client, 200, self._on_payload(None))
                return
            _http_json(client, 404, {"status": "not_found"})
        except (OSError, ValueError, TimeoutError):
            pass
        finally:
            ack_stop.set()
            with self._clients_lock:
                if client in self._clients:
                    self._clients.remove(client)
            try:
                client.close()
            except OSError:
                pass

    def _socket_loop(self, client):
        fragments = []
        fragment_opcode = None
        while not self._stop.is_set():
            try:
                frame = _read_frame(client)
            except socket.timeout:
                continue
            except (ConnectionError, OSError, ValueError):
                break
            if frame is None:
                continue
            fin, opcode, payload = frame
            if opcode == 0x8:
                _send_frame(client, 0x8, payload[:2])
                break
            if opcode == 0x9:
                _send_frame(client, 0xA, payload)
                continue
            if opcode == 0xA:
                continue
            if opcode in (0x1, 0x2):
                fragments = [payload]
                fragment_opcode = opcode
            elif opcode == 0x0 and fragments:
                fragments.append(payload)
            else:
                continue
            if not fin:
                continue
            message = b"".join(fragments)
            fragments = []
            if fragment_opcode != 0x1:
                continue
            try:
                body = json.loads(message.decode("utf-8"))
                if not isinstance(body, dict):
                    raise ValueError("target is not an object")
                self._on_payload(body)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                _send_text(client, json.dumps({"status": "error", "detail": "invalid json"}))


def _read_headers(sock):
    data = b""
    while b"\r\n\r\n" not in data:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            if not data:
                return None
            continue
        if not chunk:
            return None
        data += chunk
        if len(data) > 16384:
            raise ValueError("HTTP header is too large")
    head = data.split(b"\r\n\r\n", 1)[0].decode("iso-8859-1")
    lines = head.split("\r\n")
    method, path, _version = lines[0].split(" ", 2)
    fields = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        fields[name.strip().lower()] = value.strip()
    return {"method": method.upper(), "path": path, "fields": fields}


def _is_websocket(header):
    fields = header["fields"]
    upgrade = fields.get("upgrade", "")
    connection = fields.get("connection", "")
    return (
        "websocket" in upgrade.lower()
        and "upgrade" in connection.lower()
        and fields.get("sec-websocket-key")
    )


def _handshake(sock, header):
    accept = accept_key(header["fields"]["sec-websocket-key"])
    response = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )
    sock.sendall(response.encode("ascii"))


def _http_json(sock, status, payload):
    reason = "OK" if status == 200 else "Not Found"
    body = json.dumps(payload).encode("utf-8")
    response = (
        f"HTTP/1.1 {status} {reason}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii")+body
    sock.sendall(response)


def _read_exact(sock, size):
    data = b""
    while len(data) < size:
        _quickack(sock)
        chunk = sock.recv(size-len(data))
        if not chunk:
            raise ConnectionError("websocket closed")
        data += chunk
    return data


def _read_frame(sock):
    sock.settimeout(0.2)
    try:
        header = _read_exact(sock, 2)
    except socket.timeout:
        return None
    sock.settimeout(2.0)
    first, second = header[0], header[1]
    opcode = first & 0x0F
    fin = bool(first & 0x80)
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _read_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _read_exact(sock, 8))[0]
    if length > _MAX_FRAME:
        raise ValueError("websocket frame is too large")
    mask = _read_exact(sock, 4) if second & 0x80 else None
    payload = _read_exact(sock, length) if length else b""
    if mask is not None:
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    _quickack(sock)
    return fin, opcode, payload


def _prepare_tcp(sock):
    """Deliver small pose packets immediately instead of waiting out Nagle and delayed ACK."""
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
    except OSError:
        pass
    _quickack(sock)


def _quickack(sock):
    quickack = getattr(socket, "TCP_QUICKACK", 12)
    try:
        sock.setsockopt(socket.IPPROTO_TCP, quickack, 1)
    except OSError:
        pass


def _keep_quickack(sock, stop):
    """Quick ACK resets after each packet. A late read otherwise locks the sender near 18 Hz."""
    while not stop.is_set():
        try:
            _quickack(sock)
        except OSError:
            return
        stop.wait(0.002)


def _send_frame(sock, opcode, payload):
    data = bytes(payload)
    header = bytearray([0x80 | (opcode & 0x0F)])
    length = len(data)
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(127)
        header.extend(struct.pack("!Q", length))
    sock.sendall(bytes(header)+data)


def _send_text(sock, text):
    _send_frame(sock, 0x1, text.encode("utf-8"))
