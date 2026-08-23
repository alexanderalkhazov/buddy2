"""DuckDB connection and schema. Single file, opened lazily per call."""

from __future__ import annotations

import contextlib
from typing import Iterator

import duckdb

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker   VARCHAR,
    date     TIMESTAMP,
    open     DOUBLE,
    high     DOUBLE,
    low      DOUBLE,
    close    DOUBLE,
    volume   DOUBLE,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS fetch_log (
    ticker     VARCHAR,
    kind       VARCHAR,
    fetched_at TIMESTAMP,
    PRIMARY KEY (ticker, kind)
);

CREATE TABLE IF NOT EXISTS news (
    ticker       VARCHAR,
    url          VARCHAR,
    title        VARCHAR,
    source       VARCHAR,
    published_at TIMESTAMP,
    summary      VARCHAR,
    sentiment    VARCHAR,
    PRIMARY KEY (ticker, url)
);

CREATE TABLE IF NOT EXISTS conversation (
    id         BIGINT,
    role       VARCHAR,
    content    VARCHAR,
    created_at TIMESTAMP
);

CREATE SEQUENCE IF NOT EXISTS conversation_id_seq;

CREATE TABLE IF NOT EXISTS notes (
    key        VARCHAR PRIMARY KEY,
    value      VARCHAR,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS positions (
    id         BIGINT,
    ticker     VARCHAR,
    shares     DOUBLE,
    cost_basis DOUBLE,   -- price paid per share
    opened_at  TIMESTAMP,
    note       VARCHAR
);

CREATE SEQUENCE IF NOT EXISTS positions_id_seq;

CREATE TABLE IF NOT EXISTS watchlist (
    ticker    VARCHAR PRIMARY KEY,
    added_at  TIMESTAMP
);
"""


@contextlib.contextmanager
def connect() -> Iterator[duckdb.DuckDBPyConnection]:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.DB_PATH))
    try:
        con.execute(_SCHEMA)
        yield con
    finally:
        con.close()
