"""Pulling challenge data and pushing submissions to the team bucket.

Credentials are never stored in this repo. They come from either:

  * the environment -- AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or
  * the JSON blob the OpenSky console issues, kept outside the tree
    (default ~/.opensky/object_store_creds.json, override with
    TAXIOUT_S3_CREDENTIALS).

The console at https://s3-console.opensky-network.org is a web UI reached
by SSO and is NOT the S3 API endpoint; the API lives at
https://s3.opensky-network.org, which is what the credentials authenticate
against. The store is MinIO, so it needs path-style addressing -- virtual
host style resolves per-bucket subdomains that do not exist.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import data

TEAM = "gentle-octopus"
BUCKET = f"prc-2026-{TEAM}"

# The challenge data is in a bucket of its own, readable by every team.
DATASET_BUCKET = "prc-2026-datasets"

DEFAULT_ENDPOINT = "https://s3.opensky-network.org"
ENDPOINT_VAR = "TAXIOUT_S3_ENDPOINT"
CREDENTIALS_VAR = "TAXIOUT_S3_CREDENTIALS"
DEFAULT_CREDENTIALS = Path.home() / ".opensky" / "object_store_creds.json"


def submission_name(version: int) -> str:
    """The only filename the scoring pipeline accepts.

    Per the provisioning email: `<team>_v<N>.parquet`. A submission whose name
    does not match is dropped without a result file and without an error, so
    this is the one string in the project worth centralising.
    """
    if version < 1:
        raise ValueError(f"submission version must be >= 1, got {version}")
    return f"{TEAM}_v{version}.parquet"


def _credentials() -> tuple[str, str]:
    """Access key and secret, from the environment or the console's JSON."""
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if key and secret:
        return key, secret

    path = Path(os.environ.get(CREDENTIALS_VAR, DEFAULT_CREDENTIALS))
    if not path.exists():
        raise RuntimeError(
            f"no credentials: set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, or put "
            f"the console's JSON at {path} (or point {CREDENTIALS_VAR} at it)."
        )
    blob = json.loads(path.read_text(encoding="utf-8"))
    try:
        return blob["accessKey"], blob["secretKey"]
    except KeyError as error:
        raise RuntimeError(f"{path} has no {error.args[0]!r} field") from error


def _client():
    try:
        import boto3
        from botocore.config import Config
    except ModuleNotFoundError as error:  # pragma: no cover - dependency hint
        raise RuntimeError("boto3 is required: pip install -r pipeline/requirements.txt") from error

    key, secret = _credentials()
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get(ENDPOINT_VAR, DEFAULT_ENDPOINT),
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def list_objects(bucket: str = BUCKET, prefix: str = "") -> list[tuple[str, int]]:
    """Every object under `prefix`, as (key, size in bytes)."""
    client = _client()
    paginator = client.get_paginator("list_objects_v2")
    found: list[tuple[str, int]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for entry in page.get("Contents", []):
            found.append((entry["Key"], entry["Size"]))
    return found


def download(key: str, destination: Path, bucket: str = BUCKET) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(bucket, key, str(destination))
    return destination


def pull_dataset(
    bucket: str = DATASET_BUCKET, data_dir: Path = data.DATA_DIR, prefix: str = ""
) -> list[Path]:
    """Downloads every parquet under `prefix`, skipping ones already local.

    The challenge dataset lives in `prc-2026-datasets`, separate from the
    team's own submission bucket, which holds only what we upload.
    """
    pulled: list[Path] = []
    for key, size in list_objects(bucket, prefix):
        if not key.endswith(".parquet"):
            continue
        destination = data_dir / Path(key).name
        if destination.exists() and destination.stat().st_size == size:
            continue
        pulled.append(download(key, destination, bucket))
    return pulled


def upload_submission(path: Path, version: int, bucket: str = BUCKET) -> str:
    """Uploads `path` under the mandated submission name and returns the key."""
    if not path.exists():
        raise FileNotFoundError(path)
    key = submission_name(version)
    _client().upload_file(str(path), bucket, key)
    return key


FINAL_NAME = f"{TEAM}_final.parquet"


def upload_final(path: Path, bucket: str = BUCKET) -> str:
    """Uploads the final-phase submission, `<team>_final.parquet`.

    The ranking page asks for a unique final submission, so this refuses to
    overwrite one already in the bucket.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    if any(Path(key).name == FINAL_NAME for key, _ in list_objects(bucket)):
        raise FileExistsError(f"{FINAL_NAME} is already in {bucket}")
    _client().upload_file(str(path), bucket, FINAL_NAME)
    return FINAL_NAME


def next_version(bucket: str = BUCKET) -> int:
    """One past the highest version already in the bucket.

    Overwriting an existing submission would lose its result file, so the
    version number is derived from the bucket rather than tracked locally
    where it could drift.
    """
    highest = 0
    for key, _ in list_objects(bucket):
        name = Path(key).name
        if not (name.startswith(f"{TEAM}_v") and name.endswith(".parquet")):
            continue
        digits = name[len(f"{TEAM}_v") : -len(".parquet")]
        if digits.isdigit():
            highest = max(highest, int(digits))
    return highest + 1
