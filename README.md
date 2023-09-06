# StableDiffusionUniPipeline

👏Thanks for the work of https://github.com/hellojixian/StableDiffusionParallelPipeline


## Introduction


1. This code is mainly based on the **diffusers Python library** 🤗
2. Merging the **txt2img**(StableDiffusionPipeline) and **img2img**(StableDiffusionImg2ImgPipeline) in one pipeline.
3. Support **Double-GPUs** and **Single-GPU parallel acceleration**


## How it works

### 1. Prepare
```shell 
# 1. Install diffusers
pip install diffusers==0.20.2

# 2. Move .py to diffusers install path
mv pipeline_stable_diffusion_uni.py xxx/diffusers/pipelines/stable_diffusion/
mv pipeline_stable_diffusion_uni_parallel.py xxx/diffusers/pipelines/stable_diffusion/

## 3. Edit __init__.py
vim xxx/diffusers/pipelines/stable_diffusion/__init__.py
#line:210 + StableDiffusionUniPipeline
#line:211 + StableDiffusionUniParallelPipeline

vim xxx/diffusers/pipelines/stable_diffusion/pipelines/__init__.py
#line:110 + StableDiffusionUniPipeline
#line:111 + StableDiffusionUniParallelPipeline

vim xxx/diffusers/pipelines/stable_diffusion/pipelines/stable_diffusion/__init__.py
#line64 + from .pipeline_stable_diffusion_uni import StableDiffusionUniPipeline
#line64 + from .pipeline_stable_diffusion_uni_parallel import StableDiffusionUniParallelPipeline

```

### 2. Use

🚀 Run [benchmark_uni.ipynb](benchmark_uni.ipynb)

#### StableDiffusionUniPipeline

```python
import torch
from PIL import Image
from diffusers import StableDiffusionUniPipeline

prompt = "a photo of an astronaut riding a horse on mars"
input_imge = Image.open("./input.png")

pipe = StableDiffusionUniPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")
        
# txt2img
image = pipe(prompt).images[0]
        
# img2img
image = pipe(prompt, input_imge).images[0]
```

#### StableDiffusionUniParallelPipeline

```python
import torch
from PIL import Image
from diffusers import StableDiffusionUniParallelPipeline


prompt = "a photo of an astronaut riding a horse on mars"
input_imge = Image.open("./input.png")
        
# Double-GPUs parallel
pipe = StableDiffusionUniParallelPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
image = pipe(prompt).images[0] # txt2img
image = pipe(prompt, input_imge).images[0] # img2img

# Single-GPU parallel
pipe = StableDiffusionUniParallelPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", single_gpu_parallel=True)
image = pipe(prompt).images[0] # txt2img
image = pipe(prompt, input_imge).images[0] # img2img
```

## Feature

1. Improve code and GPUs reusability
2. Accelerate inference performance

#### Benchmark (GPU: 12GB x2)

|  Pipeline  | Time  |
|  ----   | ----  |
| StableDiffusionPipeline | 6.8s |
| StableDiffusionUniPipeline(Only-Merging)  | 6.8s |
| StableDiffusionUniParallelPipeline(Single-GPU)  | 3.4s |
| StableDiffusionUniParallelPipeline(Double-GPUs) | **1.8s** |
