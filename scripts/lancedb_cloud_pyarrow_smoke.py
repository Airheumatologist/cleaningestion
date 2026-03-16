#!/usr/bin/env python3
"""LanceDB Cloud smoke writer using PyArrow RecordBatch input."""

from __future__ import annotations

import argparse
import os
import random

import lancedb
import pyarrow as pa


def make_batches(num_batches: int, rows_per_batch: int):
    for _ in range(num_batches):
        vectors = []
        items = []
        prices = []
        for j in range(rows_per_batch):
            vectors.append([random.random() * 10.0, random.random() * 10.0])
            items.append("foo" if j % 2 == 0 else "bar")
            prices.append(random.random() * 20.0)

        yield pa.RecordBatch.from_arrays(
            [
                pa.array(vectors, type=pa.list_(pa.float32(), 2)),
                pa.array(items, type=pa.utf8()),
                pa.array(prices, type=pa.float32()),
            ],
            ["vector", "item", "price"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LanceDB Cloud PyArrow smoke writer")
    parser.add_argument("--uri", default=os.getenv("LANCEDB_URI", ""), help="LanceDB URI")
    parser.add_argument("--api-key", default=os.getenv("LANCEDB_API_KEY", ""), help="LanceDB API key")
    parser.add_argument("--region", default=os.getenv("LANCEDB_REGION", ""), help="LanceDB region")
    parser.add_argument("--table", default="my_table3", help="Target table name")
    parser.add_argument("--batches", type=int, default=100, help="Number of batches")
    parser.add_argument("--rows-per-batch", type=int, default=2, help="Rows per batch")
    args = parser.parse_args()

    if not args.uri:
        raise ValueError("LANCEDB_URI not provided. Set env var or pass --uri.")

    connect_kwargs = {}
    if args.uri.startswith("db://"):
        if not args.api_key:
            raise ValueError("LANCEDB_API_KEY not provided for LanceDB Cloud URI.")
        connect_kwargs["api_key"] = args.api_key
        if args.region:
            connect_kwargs["region"] = args.region
    db = lancedb.connect(args.uri, **connect_kwargs)
    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), 2)),
            pa.field("item", pa.utf8()),
            pa.field("price", pa.float32()),
        ]
    )

    table = db.create_table(
        args.table,
        make_batches(args.batches, args.rows_per_batch),
        schema=schema,
        mode="overwrite",
    )

    rows = table.count_rows()
    sample = table.head(3).to_pylist()
    print(f"Smoke write complete: table={args.table} rows={rows}")
    print(f"Sample rows: {sample}")


if __name__ == "__main__":
    main()
