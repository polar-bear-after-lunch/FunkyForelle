"""Connector interface.

To add a data source:

1. Create ``connectors/<name>.py`` with a class that subclasses ``Connector``.
2. Implement ``fetch(stations, start, end)`` returning the tidy schema
   (use ``schema.validate`` before returning).
3. Register it in ``connectors/__init__.py``.
4. Add an offline test with a recorded sample response in ``tests/``.
5. Add a row to ``docs/data-sources.md`` (licence, attribution, access).
"""
from __future__ import annotations

import abc
import datetime as dt
from collections.abc import Iterable

import pandas as pd
import requests

USER_AGENT = "FunkyForelle/0.1 (+https://github.com/polar-bear-after-lunch/FunkyForelle)"


class Connector(abc.ABC):
    """Base class for all data-source connectors."""

    #: short machine name, written to the ``source`` column
    name: str = "base"
    #: licence / terms of use and required attribution
    licence: str = ""
    attribution: str = ""

    def __init__(self, session: requests.Session | None = None, timeout: float = 30.0):
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.timeout = timeout

    @abc.abstractmethod
    def fetch(
        self,
        stations: Iterable[str],
        start: dt.date | str,
        end: dt.date | str,
    ) -> pd.DataFrame:
        """Return observations for ``stations`` between ``start`` and ``end`` (inclusive)."""

    # helpers -------------------------------------------------------------
    def _get(self, url: str, **params) -> requests.Response:
        resp = self.session.get(url, params=params or None, timeout=self.timeout)
        resp.raise_for_status()
        return resp


def as_date(x: dt.date | str) -> dt.date:
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    return dt.date.fromisoformat(str(x))
