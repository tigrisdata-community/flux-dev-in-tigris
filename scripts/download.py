from multiprocessing import Pool
from typing import Generator, Iterable, List
from urllib.parse import urlparse

import os
import boto3


def batcher(iterable: Iterable, batch_size: int) -> Generator[List, None, None]:
    """Batch an iterator. The last item might be of smaller len than batch_size.

    Args:
        iterable (Iterable): Any iterable that should be batched
        batch_size (int): Len of the generated lists

    Yields:
        Generator[List, None, None]: List of items in iterable
    """
    batch = []
    counter = 0
    for i in iterable:
        batch.append(i)
        counter += 1
        if counter % batch_size == 0:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


def download_batch(batch) -> int:
    s3 = boto3.client("s3")
    n = 0
    for line in batch:
        line, prefix, destdir = line
        url = urlparse(line)
        url_path = url.path.lstrip("/")

        folder, basename = os.path.split(url_path)

        dir = os.path.join(destdir, folder)
        os.makedirs(dir, exist_ok=True)
        filepath = os.path.join(dir, basename)

        print(f"{line} -> {filepath}")
        s3.download_file(url.netloc, url_path, filepath)
        n += 1
    return n


def copy_from_tigris(
        model_name: str = os.getenv("MODEL_PATH", "flux-1.dev"),
        bucket_name: str = os.getenv("MODEL_BUCKET_NAME", "cipnahubakfu"),
        destdir: str = ".",
        n_cpus: int = os.cpu_count()
    ):
    """Copy files from Tigris to the destination folder. This will be done in parallel.

    Args:
        model_name (str): model name / path to the model in the bucket. Defaults to envvar $MODEL_PATH.
        bucket_name (str): Tigris bucket to query. Defaults to envvar $BUCKET_NAME.
        destdir (str): path to store the files.
        n_cpus (int): number of simultaneous batches. Defaults to the number of cpus in the computer.

    Assumptions:
      * Every folder with a model in it is "terminal", or there are no additional models in subfolders of that folder
    """
    if not model_name.endswith("/"):
        model_name = f"{model_name}/"

    s3 = boto3.client("s3")
    model_files_resp = s3.list_objects_v2(Bucket=bucket_name, Prefix=model_name)

    model_files = [ (f"s3://{bucket_name}/{x['Key']}", model_name, destdir) for x in model_files_resp["Contents"] ]
    print(f"using {n_cpus} cpu cores for downloads")
    n_cpus = min(len(model_files), n_cpus)
    batch_size = len(model_files) // n_cpus
    with Pool(processes=n_cpus) as pool:
        for n in pool.imap_unordered(
            download_batch, batcher(model_files, batch_size)
        ):
            pass

    return os.path.join(destdir, model_name)
