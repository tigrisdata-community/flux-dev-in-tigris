# Storing Model Weights in Tigris

The most common way to deploy AI models in production is by using “serverless”
inference. This means that every time you get a request, you don’t know what
state the underlying hardware is in. You don’t know if you have your models
cached, and in the worst case you need to do a cold start and download your
model weights from scratch.

A couple fixable problems arise when running your models on serverless or any
frequently changing infrastructure:

- Model distribution that's not optimized for latency causes needless GPU idle
  time as the model weights are downloaded to the machine on cold start. Tigris
  behaves like a content delivery network by default and is designed for low
  latency, saving idle time on cold start.
- Compliance restrictions like data sovereignty and GDPR increase complexity
  quickly. Tigris makes regional restrictions a one-line configuration, guide
  [here](https://www.tigrisdata.com/docs/objects/object_regions/).
- Reliance on third party caches for distributing models creates an upstream
  dependency and leaves your system vulnerable to downtime. Tigris guarantees
  99.99% availability with
  [public availability data](https://www.tigrisdata.com/blog/availability-metrics-public/).

## Usecase

You can put AI model weights into Tigris so that they are cached and fast no
matter where you’re inferencing from. This allows you to have cold starts be
faster and you can take advantage of Tigris'
[globally distributed architecture](/docs/overview/), enabling your workloads to
start quickly no matter where they are in the world.

For this example, we’ll set up
[SDXL Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) by ByteDance
for inference with the weights stored in Tigris. Here's what you need to do:

- Prepare and upload the model to Tigris
- Create a restricted access key for model runners
- Run inference somewhere

Download the `sdxl-in-tigris` template from GitHub:

```text
git clone https://github.com/tigrisdata-community/sdxl-in-tigris
```

<details>
<summary>Prerequisite tools</summary>

In order to run this example locally, you need these tools installed:

- Python 3.11
- pipenv
- The AWS CLI

Also be sure to configure the AWS CLI for use with Tigris:
[Configuring the AWS CLI](/docs/sdks/s3/aws-cli/).

To build a custom variant of the image, you need these tools installed:

- Mac/Windows:
  [Docker Desktop app](https://www.docker.com/products/docker-desktop/),
  alternatives such as Podman Desktop will not work.
- Linux: Docker daemon, alternatives such as Podman will not work.
- [Replicate's cog tool](https://github.com/replicate/cog)
- [jq](https://jqlang.github.io/jq/)

To install all of the tool depedencies at once, clone the template repo and run
`brew bundle`.

</details>

Create two new buckets:

1. One bucket will be for generated images, it’ll be called `generated-images`
   in this article
2. One bucket will be for storing models, it’ll be called `model-storage` in
   this article

```text
aws s3 create-bucket --acl private generated-images
aws s3 create-bucket --acl private model-storage
```

Both of these buckets should be private.

Then activate the virtual environment with `pipenv shell` and install the
dependencies for uploading a model:

```text
pipenv shell --python 3.11
pip install -r requirements.txt
```

Then run the `prepare_model` script to massage and upload a Stable Diffusion XL
model or finetune to Tigris:

```text
python scripts/prepare_model.py ByteDance/SDXL-Lightning model-storage
```

:::info

Want differently styled images? Try finetunes like
[Kohaku XL](https://huggingface.co/KBlueLeaf/Kohaku-XL-Zeta)! Pass the Hugging
Face repo name to the `prepare_model` script like this:

```text
python scripts/prepare_model.py KBlueLeaf/Kohaku-XL-Zeta model-storage
```

:::

This will take a bit to run, depending on your internet connection speed, hard
drive speed, and the current phase of the moon. While it’s running, head to the
Tigris console and create a new access key. Don't assign any permissions to it.

### Create a restricted access key for model runners

Copy the access key ID and secret access keys into either your notes or a
password manager, you will not be able to see them again. These credentials will
be used later to deploy your app in the cloud. This keypair will be referred to
as the `workload-keypair` in this tutorial.

Open `iam/model-runner.json` in your text editor. Change all references for
`model-storage` and `generated-images` to the buckets you created earlier.

Then export this variable to make IAM changes in Tigris:

```text
AWS_ENDPOINT_URL_IAM=https://fly.iam.storage.tigris.dev
```

Create an IAM policy based on the document you edited:

```text
aws iam create-policy --policy-name sdxl-runner --policy-document file://./iam/model-runner.json
```

Copy down the ARN in the output, it should look something like this:

```text
arn:aws:iam::flyio_hunter2hunter2hunter2:policy/sdxl-runner
```

Attach it to the token you just created:

```text
aws iam attach-user-policy \
  --policy-arn arn:aws:iam::flyio_hunter2hunter2:policy/sdxl-runner \
  --user-name tid_workload_keypair_access_key_id
```

### Running inference

<details>
<summary>Optional: building your own image</summary>

In order to deploy this, you need to build the image with the cog tool. Log into
a Docker registry and run this command to build and push it:

```text
cog push your-docker-username/sdxl-tigris --use-cuda-base-image false
```

</details>

You can now use it with your GPU host of choice as long as it support at least
Cuda 12.1 and has at least 12 GB of video memory.

This example is configured with environment variables. Set the following
environment variables in your deployments:

|             Envvar name | Value                                                  |
| ----------------------: | :----------------------------------------------------- |
|     `AWS_ACCESS_KEY_ID` | The access key ID from the workload keypair            |
| `AWS_SECRET_ACCESS_KEY` | The secret access key from the workload keypair        |
|   `AWS_ENDPOINT_URL_S3` | `https://fly.storage.tigris.dev`                       |
|            `AWS_REGION` | `auto`                                                 |
|            `MODEL_PATH` | `ByteDance/SDXL-Lightning`                             |
|     `MODEL_BUCKET_NAME` | `model-storage` (replace with your own bucket name)    |
|    `PUBLIC_BUCKET_NAME` | `generated-images` (replace with your own bucket name) |

You can run this on any platform you want that has the right GPUs, and Skypilot
makes this easy. [Skypilot](https://skypilot.readthedocs.io/en/latest/docs/index.html)
is a tool that lets you route GPU compute to the cheapest possible locale based
on your requirements. The same configuration lets you control AWS, Azure, Google
Cloud, Oracle Cloud, Kubernetes, Runpod, Fluidstack, or more. For more information
about Skypilot, check out
[their documentation](https://skypilot.readthedocs.io/en/latest/docs/index.html).

To get started, you'll need to install Skypilot
[following their directions](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html).
Be sure to have [Conda](https://anaconda.org/anaconda/conda) installed.

You will need to configure your cloud of choice for this example. See
[Skypilot's documentation](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html#cloud-account-setup)
on how to do this. We have tested this against a few clouds:

- [AWS](https://aws.amazon.com/)
- [Lambda](https://lambdalabs.com/)
- [Runpod](https://www.runpod.io/)

However the other providers should work fine.

## Customizing the `skypilot.yaml` file

Open `skypilot.yaml` in your favorite text editor. Customize the environment
variables in the `envs:` key:

```yaml
envs:
  # Tigris config
  AWS_ACCESS_KEY_ID: tid_AzureDiamond # workload access key ID
  AWS_SECRET_ACCESS_KEY: tsec_hunter2 # workload secret access key
  AWS_ENDPOINT_URL_S3: https://fly.storage.tigris.dev
  AWS_REGION: auto

  # Bucket names
  MODEL_BUCKET_NAME: model-storage
  PUBLIC_BUCKET_NAME: generated-images

  # Model to load
  MODEL_PATH: ByteDance/SDXL-Lightning
```

|             Envvar name | Value                                                  |
| ----------------------: | :----------------------------------------------------- |
|     `AWS_ACCESS_KEY_ID` | The access key ID from the workload keypair            |
| `AWS_SECRET_ACCESS_KEY` | The secret access key from the workload keypair        |
|   `AWS_ENDPOINT_URL_S3` | `https://fly.storage.tigris.dev`                       |
|            `AWS_REGION` | `auto`                                                 |
|            `MODEL_PATH` | `ByteDance/SDXL-Lightning`                             |
|     `MODEL_BUCKET_NAME` | `model-storage` (replace with your own bucket name)    |
|    `PUBLIC_BUCKET_NAME` | `generated-images` (replace with your own bucket name) |

## Launching it in a cloud

Run `sky serve up` to start the image in a cloud:

```text
sky serve up skypilot.yaml -n sdxl
```

Wait a few minutes for everything to converge, and then you can use the endpoint
URL to poke it:

```text
⚙︎ Service registered.

Service name: sdxl
Endpoint URL: 3.84.60.169:30001
```

:::note

You can run `sky serve status` to find out if your endpoint is ready:

```text
$ sky serve status
<...>
Service Replicas
SERVICE_NAME  ID  VERSION  ENDPOINT                  LAUNCHED     RESOURCES                   STATUS  REGION
sdxl          1   1        http://69.30.85.69:22112  47 secs ago  1x RunPod({'RTXA4000': 1})  READY   CA
```

:::

Finally, run a test generation with this curl command:

```text
curl "http://ip:port/predictions/$(uuidgen)" \
  -X PUT \
  -H "Content-Type: application/json" \
  --data-binary '{
    "input": {
        "prompt": "The space needle in Seattle, best quality, masterpiece",
        "aspect_ratio": "1:1",
        "guidance_scale": 3.5,
        "num_inference_steps": 4,
        "max_sequence_length": 512,
        "output_format": "png",
        "num_outputs": 1
    }
}'
```

If all goes well, you should get an image like this:

![The word 'success' in front of the Space Needle](./success.webp)

You can destroy the machine with this command:

```text
sky serve down sdxl
```
