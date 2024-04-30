import os
import shutil
import mimetypes
import json
import random
from PIL import Image
from typing import List
from cog import BasePredictor, Input, Path
from helpers.comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"

SAMPLERS = [
    "euler",
    "euler_ancestral",
    "heun",
    "heunpp2",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
]

SDXL_WEIGHTS = [
    "albedobaseXL_v13.safetensors",
    "albedobaseXL_v21.safetensors",
    "CinematicRedmond.safetensors",
    "copaxTimelessxlSDXL1_v8.safetensors",
    "dreamshaperXL_alpha2Xl10.safetensors",
    "epicrealismXL_v10.safetensors",
    "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
    "juggernautXL_v8Rundiffusion.safetensors",
    "pixlAnimeCartoonComic_v10.safetensors",
    "proteus_v02.safetensors",
    "ProteusV0.4.safetensors",
    "RealVisXL_V2.0.safetensors",
    "RealVisXL_V3.0.safetensors",
    "RealVisXL_V4.0.safetensors",
    "rundiffusionXL_beta.safetensors",
    "sd_xl_base_1.0.safetensors",
    "sd_xl_base_1.0_0.9vae.safetensors",
    "starlightXLAnimated_v3.safetensors",
]

mimetypes.add_type("image/webp", ".webp")


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

    def cleanup(self):
        self.comfyUI.clear_queue()
        for directory in [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    def update_workflow(self, workflow, **kwargs):
        workflow["4"]["inputs"]["ckpt_name"] = kwargs["checkpoint"]
        workflow["22"]["inputs"]["sampler_name"] = kwargs["sampler_name"]
        workflow["25"]["inputs"]["steps"] = kwargs["steps"]
        workflow["6"]["inputs"]["text"] = kwargs["prompt"]
        workflow["19"]["inputs"]["text"] = f"nsfw, nude, {kwargs['negative_prompt']}"

        empty_latent_image = workflow["5"]["inputs"]
        empty_latent_image["width"] = kwargs["width"]
        empty_latent_image["height"] = kwargs["height"]
        empty_latent_image["batch_size"] = kwargs["num_outputs"]

        sampler = workflow["21"]["inputs"]
        sampler["noise_seed"] = kwargs["seed"]
        sampler["cfg_scale"] = kwargs["guidance_scale"]

    def log_and_collect_files(self, directory, prefix=""):
        files = []
        for f in os.listdir(directory):
            if f == "__MACOSX":
                continue
            path = os.path.join(directory, f)
            if os.path.isfile(path):
                print(f"{prefix}{f}")
                files.append(Path(path))
            elif os.path.isdir(path):
                print(f"{prefix}{f}/")
                files.extend(self.log_and_collect_files(path, prefix=f"{prefix}{f}/"))
        return files

    def predict(
        self,
        prompt: str = Input(default="a photo of an astronaut riding a unicorn"),
        negative_prompt: str = Input(
            description="The negative prompt to guide image generation.",
            default="",
        ),
        checkpoint: str = Input(
            choices=SDXL_WEIGHTS,
            description="The SDXL model to use for generation",
            default="albedobaseXL_v21.safetensors",
        ),
        width: int = Input(default=1024),
        height: int = Input(default=1024),
        num_outputs: int = Input(
            description="Number of outputs", ge=1, le=10, default=1
        ),
        sampler_name: str = Input(
            choices=SAMPLERS,
            default="dpmpp_2m_sde_gpu",
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance",
            default=7.5,
            ge=0,
            le=30.0,
        ),
        steps: int = Input(
            description="Number of diffusion steps. (A minimum of 10 with AYS)",
            ge=10,
            le=100,
            default=10,
        ),
        seed: int = Input(default=None),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality of the output images, from 0 to 100. 100 is best quality, 0 is lowest quality.",
            default=80,
            ge=0,
            le=100,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.cleanup()

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print(f"Random seed set to: {seed}")

        with open("ays_api.json", "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            checkpoint=checkpoint,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_outputs=num_outputs,
            sampler_name=sampler_name,
            seed=seed,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        files = []
        output_directories = [OUTPUT_DIR]

        for directory in output_directories:
            print(f"Contents of {directory}:")
            files.extend(self.log_and_collect_files(directory))

        if output_quality < 100 or output_format in ["webp", "jpg"]:
            optimised_files = []
            for file in files:
                if file.is_file() and file.suffix in [".jpg", ".jpeg", ".png"]:
                    image = Image.open(file)
                    optimised_file_path = file.with_suffix(f".{output_format}")
                    image.save(
                        optimised_file_path,
                        quality=output_quality,
                        optimize=True,
                    )
                    optimised_files.append(optimised_file_path)
                else:
                    optimised_files.append(file)

            files = optimised_files

        return files
