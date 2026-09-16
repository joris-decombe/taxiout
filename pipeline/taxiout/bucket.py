"""Pulling challenge data and pushing submissions to the team bucket.

Credentials are never read from a file in this repo and never passed as
arguments. Set them in the environment before running:

    TAXIOUT_S3_ENDPOINT   the S3 API endpoint (NOT the console URL)
    AWS_ACCESS_KEY_ID     access key generated in the OpenSky S3 console
    AWS_SECRET_ACCESS_KEY its secret

The console at https://s3-console.opensky-network.org is a web UI reached by
SSO; it is not an S3 API endpoint, so `TAXIOUT_S3_ENDPOINT` has to be the
address the console shows under its access-key page. It is left unset by
default deliberately -- guessing it wrong produces a confusing TLS error
rather than an honest "you have not configured this yet".
"""

from __future__ import annotations

import os
from pathlib import Path

from . import data

TEAM = "gentle-octopus"
BUCKET = f"prc-2026-{TEAM}"
ENDPOINT_VAR = "TAXIOUT_S3_ENDPOINT"


def submission_name(version: int) -> str:
    """The only filename the scoring pipeline accepts.

    Per the provisioning email: `<team>_v<N>.parquet`. A submission whose name
    does not match is dropped without a result file and without an error, so
    this is the one string in the project worth centralising.
    """
    if version < 1:
        raise ValueError(f"submission version must be >= 1, got {version}")
    return f"{TEAM}_v{version}.parquet"


def _client():
    try:
        import boto3
    except ModuleNotFoundError as error:  # pragma: no cover - dependency hint
        raise RuntimeError("boto3 is required: pip install -r pipeline/requirements.txt") from error

    endpoint = os.environ.get(ENDPOINT_VAR)
    if not endpoint:
        raise RuntimeError(
            f"{ENDPOINT_VAR} is unset. Set it to the S3 API endpoint from the "
            "OpenSky console's access-key page, along with AWS_ACCESS_KEY_ID "
            "and AWS_SECRET_ACCESS_KEY."
        )
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        raise RuntimeError("AWS_ACCESS_KEY_ID is unset -- generate a key in the OpenSky console.")

    return boto3.client("s3", endpoint_url=endpoint)


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


def pull_dataset(bucket: str, data_dir: Path = data.DATA_DIR, prefix: str = "") -> list[Path]:
    """Downloads every parquet under `prefix`, skipping ones already local.

    The challenge dataset lives in a bucket separate from the team's own
    submission bucket, so the source bucket is required rather than defaulted.
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
