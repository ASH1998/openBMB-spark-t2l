"""Build and publish the named GPU image used by Gate 1 and later stages.

    python -m modal_app.images            # build once (compiles causal-conv1d, ~15 min) and publish

The image is published as $T2L_GPU_IMAGE (default "minicpm-t2l-gpu"). Apps load it with
`modal.Image.from_name(...)`, so the CUDA compile happens once rather than per app.
Keep the package versions here in sync with modal_app/kernel_check.py, which verified them.
"""

from __future__ import annotations

import os

import modal

NAME = os.environ.get("T2L_GPU_IMAGE", "minicpm-t2l-gpu")


def gpu_image() -> modal.Image:
    return (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
        .apt_install("git", "build-essential")
        .pip_install("torch==2.10.0", "torchvision==0.25.0", "ninja", "packaging", "wheel")
        .pip_install("flash-linear-attention", "transformers>=5.21", "peft>=0.19", "accelerate", "pillow",
                     "huggingface_hub>=1.10", "safetensors", "datasets>=4.3")
        .run_commands("TORCH_CUDA_ARCH_LIST='8.0 8.6 8.9 9.0' MAX_JOBS=8 pip install --no-build-isolation causal-conv1d", gpu="A10G")
        .env({"HF_HOME": "/vol/hf"})
    )


if __name__ == "__main__":
    # Pattern from https://modal.com/docs/guide/named-images: build against a looked-up App, then publish.
    app = modal.App.lookup(os.environ.get("T2L_MODAL_APP_NAME", "minicpm-t2l") + "-image-builds", create_if_missing=True)
    img = gpu_image()
    with modal.enable_output():
        img.build(app)
        img.publish(NAME)
    print(f"published {NAME}")
