from glob import glob
from multiprocessing import Pool
from shutil import rmtree
from typing import Generator, Iterable, List
import argparse
import boto3
import os
import sys


try:
    import torch
    _ = torch
except ImportError:
    print("install torch and diffusers before proceeding, see README.md in the root of this repository")
    sys.exit(2)


HF_MODEL_NAME="ByteDance/SDXL-Lightning"
DESTDIR="local_setup"


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


def fetch_and_save_model(model_name, destdir):
    import torch
    from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
    from huggingface_hub import hf_hub_download

    dtype = torch.bfloat16
    pipe = None

    match model_name:
        case "ByteDance/SDXL-Lightning":
            from safetensors.torch import load_file
            base = "stabilityai/stable-diffusion-xl-base-1.0"
            repo = "ByteDance/SDXL-Lightning"
            ckpt = "sdxl_lightning_4step_unet.safetensors"
            unet = UNet2DConditionModel.from_config(base, subfolder="unet")
            unet.load_state_dict(load_file(hf_hub_download(repo, ckpt)))
            pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=dtype, variant="fp16")
        case _:
            pipe = StableDiffusionXLPipeline.from_pretrained(model_name, torch_dtype=dtype)
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    
    pipe.save_pretrained(destdir, safe_serialization=True)


def upload_batch(batch) -> int:
    s3 = boto3.client("s3")
    n = 0

    for line in batch:
        src, bucket, dst = line
        print(f"{src} -> s3://{bucket}/{dst}")
        s3.upload_file(src, bucket, dst)
        n += 1

    return n


def push_model(bucket_name, bucket_path, localdir, n_cpus):
    files = [
        (
            x, 
            bucket_name,
            bucket_path + x[len(localdir):],
        ) for x in glob(f"{localdir}/**/*") + glob(f"{localdir}/*.json")
    ]

    print(f"using {n_cpus} cpu cores for uploads")
    n_cpus = min(len(files), n_cpus)
    batch_size = len(files) // n_cpus

    with Pool(processes=n_cpus) as pool:
        for n in pool.imap_unordered(
            upload_batch, batcher(files, batch_size)
        ):
            pass


def main():
    parser = argparse.ArgumentParser("prepare_model", description="prepare a model for inference via Tigris")
    parser.add_argument("hf_model_name", default=HF_MODEL_NAME, help="Hugging Face repo path")
    parser.add_argument("bucket_name", help="Tigris bucket name to upload the model to")
    parser.add_argument("--destdir", default=DESTDIR, help="temporary model storage directory")
    parser.add_argument("--num_cpu", "-j", default=os.cpu_count(), help="number of uploads to do in parallel (defaults to CPU core count)")
    
    args = parser.parse_args()

    hf_model_name = args.hf_model_name
    bucket_name = args.bucket_name
    destdir = args.destdir
    num_cpu = args.num_cpu

    try:
        os.makedirs(destdir)
    except FileExistsError:
        pass
    
    print(f"""using these settings:
* hugging face model repo:      {hf_model_name}
* bucket name:                  {bucket_name}
* temporary model storage dir:  {destdir}
* parallel upload jobs:         {num_cpu}

Let's a-go!
""")
    
    if not os.path.exists("local_setup/model_index.json"):
        print(f"fetching and saving model {hf_model_name} to {destdir}...")
        fetch_and_save_model(hf_model_name, destdir)
        print("...done")
    else:
        print("model already prepared, skipping the fetch and save step")

    print(f"uploading dir {destdir} to Tigris bucket {bucket_name}")
    push_model(bucket_name, hf_model_name, destdir, num_cpu)
    print("...done")

    print(f"""

uploaded, use the following environment variables in your deployment:

  AWS_ACCESS_KEY_ID=<key from earlier>
  AWS_SECRET_ACCESS_KEY=<key from earlier>
  AWS_ENDPOINT_URL_S3=https://fly.storage.tigris.dev
  AWS_REGION=auto
  MODEL_BUCKET_NAME={bucket_name}
  MODEL_PATH={hf_model_name}
""")


if __name__ == "__main__":
    main()
