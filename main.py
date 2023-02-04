import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
import string
import random
import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline
from RealESRGAN import RealESRGAN
from appscript import app, mactypes

# todo:
# get laptop screen dimensions, are they divisible by 8?
# put all functions in a utils file
# windows support?
# run from cli w/ a cmd?
# try a different upscaling model

def get_device():
    if(torch.cuda.is_available()):
        return "cuda"
    elif(torch.backends.mps.is_available()):
        return "mps"
    else:
        return "cpu"
    
def nature_scene_prompt_seed():
    scene_close_natural_feature_seeds = ["serene lake", "peaceful river bank", "tranquil island"]
    scene_manmade_structure_seeds = ["bridge", "cathedral", "skyscraper", "pagoda", "dome", "cabin", "amphitheatre", "hut", "tent", "greenhouse", "church", "mosque", "boat", "mansion", "treehouse", "altar", "temple", "ruin", "castle", "plane", "hot air balloon", "pyramid"]
    scene_setting = ["forest", "park"]
    scene_distant_natural_feature_seeds = ["mountains", "the moon", "stars", "the sun", "a city", "village"]
    prompt_seed = f"highly detailed matte oil painting of a {random.choice(scene_close_natural_feature_seeds)} with a {random.choice(scene_manmade_structure_seeds)} set in a verdant {random.choice(scene_setting)} with beautiful huge trees, an inspiring blue sky with impressive clouds and {random.choice(scene_distant_natural_feature_seeds)} in the distance"
    return prompt_seed

def generate_prompt():
    print('Step 1: generating image prompt... ⏳')
    generator = pipeline('text-generation', model="mrm8488/bloom-560m-finetuned-sd-prompts")
    prompt = generator(nature_scene_prompt_seed(), max_length=77, num_return_sequences=1)[0]['generated_text']
    print(f"Done ✅")
    print(f"🗣️ Prompt: {prompt}")
    print("")
    return prompt

def generate_image(prompt):
    model_id = "CompVis/stable-diffusion-v1-4" # "CompVis/stable-diffusion-v1-4", "stabilityai/stable-diffusion-2"
    num_inference_steps = 75
    guidance_scale = 7.5 
    image_height = 768
    image_width = 1024

    print('Step 2: generating image... ⏳')
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token=True, cache_dir=os.getenv("cache_dir", "./models"))
    pipe = pipe.to(get_device())
    pipe.enable_attention_slicing()
    # ? kwargs not reached
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    # HuggingFace: This is a temporary workaround for a weird issue
    _ = pipe(prompt, num_inference_steps=1)
    image = pipe(prompt, guidance_scale=guidance_scale, height=image_height, width=image_width, num_inference_steps=num_inference_steps).images[0]
    print(f"Done ✅ ")
    print("")
    return image

def upscale_image(image):
    print('Step 3: upscaling image... ⏳')
    device = torch.device('cpu')
    model = RealESRGAN(device, scale=4)
    model.load_weights(f"weights/RealESRGAN_x4.pth", download=True)
    upscaled_image = model.predict(image)
    print('Done ✅ ')
    print("")
    return upscaled_image

def save_image(image, image_name, image_path):
    print('Step 4: saving image... ⏳')
    image.save(image_path)
    print(f"{image_name}.png saved 💾")
    print("")

def set_desktop_bg(image_path):
    print('Step 5: setting image to BG... ⏳ ')
    app('Finder').desktop_picture.set(mactypes.File(f"{image_path}"))
    print('Done ✅ ')
    print("")

def create_and_set_new_bg_image():
    print("")
    print("CREATING NEW BACKGROUND IMAGE 🎨")
    print("")
    image_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    image_path = f"./images/{image_name}.png"
    image_prompt = generate_prompt()
    image = generate_image(image_prompt)
    upscaled_image = upscale_image(image)
    save_image(upscaled_image,image_name, image_path)
    set_desktop_bg(image_path)
    print("")
    print("NEW BG SET 🚀 ")
    print("")

# warnings.filterwarnings("ignore")
create_and_set_new_bg_image()