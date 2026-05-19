from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import Any
import secrets

import numpy as np


@dataclass
class SessionState:
    session_id: str
    created_at: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    updated_at: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    image_rgb: np.ndarray | None = None
    image_name: str = ''
    image_id: str = ''
    gt_mask: np.ndarray | None = None

    def touch(self) -> None:
        self.updated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class SessionStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[str, SessionState] = {}

    def new(self, session_id: str | None = None) -> SessionState:
        with self._lock:
            sid = str(session_id or '').strip() or f'sr_{secrets.token_hex(6)}'
            sess = SessionState(session_id=sid)
            self._sessions[sid] = sess
            return sess

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            return self._sessions.get(str(session_id or '').strip())


store = SessionStore()
