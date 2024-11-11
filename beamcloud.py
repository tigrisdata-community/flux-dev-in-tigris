import os
from beam import Image, Volume, endpoint, Output, env


if env.is_remote():
    from predict import Predictor


image = Image(
    python_version="python3.11",
    python_packages=[
        "diffusers",
        "torch==2.2",
        "transformers==4.41.2",
        "accelerate==0.31.0",
        "sentencepiece",
        "protobuf",
        "boto3",
        "sqids",
        "numpy<2",
        "cog",
        "compel",
    ]
)


def load_models():    
    result = Predictor()
    result.setup()
    return result


@endpoint(
    name="sdxl-tigris",
    secrets=[
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "AWS_ENDPOINT_URL_S3",
        "MODEL_PATH",
        "MODEL_BUCKET_NAME",
        "PUBLIC_BUCKET_NAME",
    ],
    image=image,
    on_start=load_models,
    keep_warm_seconds=60,
    cpu=2,
    gpu="A100-40",
    volumes=[
        Volume(name="models", mount_path="/src"),
    ],
)
def generate(context, **inputs):
    predictor: Predictor = context.on_start_value

    prompt = inputs.get("prompt", None)
    negative_prompt = inputs.get("negative_prompt", "")
    aspect_ratio = inputs.get("aspect_ratio", "1:1")
    num_outputs = inputs.get("num_outputs", 1)
    guidance_scale = inputs.get("guidance_scale", 10.5)
    max_sequence_length = inputs.get("max_sequence_length", 256)
    num_inference_steps = inputs.get("num_inference_steps", 50)
    seed = inputs.get("seed", None) # can be None, it's okay
    output_format = inputs.get("output_format", "webp")
    output_quality = inputs.get("output_quality", 80)
    
    result = predictor.predict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        aspect_ratio=aspect_ratio,
        num_outputs=num_outputs,
        guidance_scale=guidance_scale,
        max_sequence_length=max_sequence_length,
        num_inference_steps=num_inference_steps,
        seed=seed,
        output_format=output_format,
        output_quality=output_quality,
    )

    return {
        "inputs": inputs,
        "result": result,
    }
