import argparse
import os
import sys
import time
import uuid

from diffusers import StableDiffusionImg2ImgPipeline
import optuna
from PIL import Image
import torch

import optuna_dashboard.preferential
from optuna_dashboard.preferential import PreferentialStudy
from optuna_dashboard.preferential.samplers.gp import PreferentialGPSampler
from optuna.trial import FrozenTrial, TrialState
from optuna.artifacts import FileSystemArtifactStore
from optuna.artifacts import upload_artifact
from optuna_dashboard.artifact import get_artifact_path
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, default=0)
args = parser.parse_args(sys.argv[1:])


torch.cuda.set_device(args.gpu_id)
device = "cuda"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
)
pipe = pipe.to(device)
pipe.enable_xformers_memory_efficient_attention()

rng = torch.Generator(device)
rng.seed()

input_img = Image.open("./diffusion_input.png").convert("RGB")
input_size = 768
input_img = input_img.resize((input_size, input_size), Image.Resampling.LANCZOS)



def generate_image(guidance_scale, strength, prompt, negative_prompt):
    num_inference_steps = 50
    images = pipe(
        prompt,
        negative_prompt=negative_prompt,
        generator=rng,
        strength=strength,
        image=input_img,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=1,
    ).images
    return images[0]

STORAGE_URL = "sqlite:///example.db"

study = optuna_dashboard.preferential.create_study(
    study_name="preferential_diffusion",
    storage=STORAGE_URL,
    sampler=PreferentialGPSampler(),
    load_if_exists=True,
    n_generate=5,
)

artifact_path = os.path.join(os.path.dirname(__file__), "artifact")
os.makedirs(artifact_path, exist_ok=True)
artifact_store = FileSystemArtifactStore(base_path=artifact_path)

optuna_dashboard.register_preference_feedback_component(study, component_type="artifact", artifact_key="image")

with tempfile.TemporaryDirectory() as tmpdir:
    while True:
        if not study.should_generate():
            time.sleep(0.1)  # Avoid busy-loop
            continue

        trial = study.ask()

        print(f"Ask: {trial.number}")

        guidance_scale = trial.suggest_categorical(
            "guidance_scale", [1, 4, 16, 50]
        )
        strength = trial.suggest_categorical(
            "strength", [0.7, 0.8, 0.9, 0.95]
        )

        prompts = ["a mascot character with two eyes and a mouth"]
        prompts.append(
            trial.suggest_categorical(
                "adjectives", ["cute", "funny", "memorable", "charming", "entertaining"]
            )
        )
        prompts.append(trial.suggest_categorical("style", ["anime", "photo", "painting", ""]))
        prompts.append(
            trial.suggest_categorical("facial-expression", ["smiling", "frowning", "grinning", ""])
        )
        prompt_str = ", ".join(prompts)
        trial.set_user_attr("prompt", prompt_str)

        negative_prompt = []
        negative_prompt.append(
            trial.suggest_categorical("negative-quality", ["unnatural", "low-quality", ""])
        )
        negative_prompt.append(
            trial.suggest_categorical("negative-adjectives", ["dull", "boring", "unfriendly", ""])
        )
        negative_prompt_str = ", ".join(negative_prompt)
        trial.set_user_attr("negative_prompt", negative_prompt_str)

        print(f"Generating image. (trial.number={trial.number})")

        image = generate_image(guidance_scale, strength, prompt_str, negative_prompt_str)
        image_path = os.path.join(tmpdir, f"sample-{trial.number}.png")
        image.save(image_path)
        print(f"Generation done. (image: {image_path})")

        artifact_id = upload_artifact(trial, image_path, artifact_store)
        trial.set_user_attr("image", artifact_id)
        print(f"Upload done. (url: {get_artifact_path(trial, artifact_id)})")



# %%