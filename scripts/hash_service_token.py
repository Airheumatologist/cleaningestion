#!/usr/bin/env python3
"""Generate bcrypt hashes for service bearer tokens."""

import argparse
import getpass

from src.service_auth import hash_service_token


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bcrypt hash for a service token")
    parser.add_argument("--token", help="Plaintext token (omit to read securely from stdin)")
    parser.add_argument("--rounds", type=int, default=12, help="bcrypt cost factor (default: 12)")
    args = parser.parse_args()

    token = args.token or getpass.getpass("Service token: ")
    if not token.strip():
        raise SystemExit("Token cannot be empty")

    print(hash_service_token(token.strip(), rounds=args.rounds))


if __name__ == "__main__":
    main()
