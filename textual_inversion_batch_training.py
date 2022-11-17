import argparse, itertools, math, os, random, PIL
import numpy as np, torch, torch.nn.functional as F, torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_scheduler
from PIL import Image
from PIL.Image import Resampling
from torchvision import transforms
from tqdm.auto import tqdm

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import fastcore.all as fc
from huggingface_hub import notebook_login
from pathlib import Path

import torchvision.transforms.functional as tf
import accelerate
import yaml
import string


torch.manual_seed(42)
print(Path.home()/'.huggingface'/'token')
if not (Path.home()/'.huggingface'/'token').exists(): notebook_login()


batch_config =  yaml.safe_load(open('batch_config.yaml','r'))

Path(batch_config['base_outpath']).mkdir(exist_ok=True)

model_nm = batch_config['model']

concept_batch = []
for batch in batch_config['batches']:
        
    batch_path = Path(batch_config['datasets_path']) / batch['name']
    batch_outpath = Path(batch_config['base_outpath']) / batch['name']
    
    # create the out path folder for this batch if it doesn't exist
    batch_outpath.mkdir(exist_ok=True)

    if (batch_path/'inits.yaml').exists():
        inits = yaml.safe_load(open(batch_path/'inits.yaml','r'))
    else:
        inits = {}
    
    
    # folders in the batch path define which template to use
    # arbitrary templates can be defined in the yaml file
    for t in batch['templates']:
        
        concepts_path = batch_path / t
        
        # folders in the template path name the concept
        for c in concepts_path.iterdir():
            if c.is_dir():
                
                # concepts should have unique names anyways so outpath
                # does not include template
                outpath = batch_outpath / c.stem
                outpath.mkdir(exist_ok=True)
                 
                concept_batch.append({
                    'name': c.stem,
                    'path': c,
                    'init': inits.get(c.stem,batch_config['default_init']),
                    'template': t,
                    'outpath': outpath
                })

print(concept_batch)


class TextualInversionDataset:
    def __init__(self, tokenizer, images, template="subject", size=512,
                 repeats=100, interpolation=Resampling.BICUBIC, flip_p=0.5, set="train", placeholder_token="*"):
        fc.store_attr()
        self.num_images = len(images)
        if set == "train": self._length = self.num_images * repeats
        self.template_type = template
        self.templates = batch_config['templates'][template]
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)
        self.last_prompt = ""

    def __len__(self): return self.num_images

    def __getitem__(self, i):
        image = self.images[i%self.num_images]
        image_path = Path(image[1])
        image = tf.to_tensor(image[0])*2-1
        
        description_file = image_path.with_name(image_path.stem+".txt")
        text = random.choice(self.templates)
        tokens = len(list(string.Formatter().parse(text)))
        if description_file.exists():
            with open(description_file) as f:
                lines = [line.rstrip() for line in f]
                description = lines[0]

            text = text.format(description=description, token=self.placeholder_token)
        else:
            if tokens == 1:
                text = text.format(self.placeholder_token)
            elif tokens == 2:
                text = text.format(description=batch_config['default_description'], token=self.placeholder_token)

        self.last_prompt = text
        ids=self.tokenizer(text, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt")
        return dict(input_ids=ids.input_ids[0], pixel_values=image)

def train_concept(concept):
    initializer_token = concept['init']
    path = concept['path'] 
    img_paths = list(path.iterdir())
    print(img_paths)
    images = []
    try:
        img_paths =  filter(lambda p: p.name != "Thumbs.db", img_paths)
    except:
        pass
    for i,p in enumerate(img_paths):
        if p.suffix != ".txt":
            images.append([Image.open(p).resize((512, 512), resample=Resampling.BICUBIC).convert("RGB"),p])

    placeholder_token = f"<{concept['name']}>"

    tokenizer = CLIPTokenizer.from_pretrained(model_nm, subfolder="tokenizer", use_auth_token=True)
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    text_encoder = CLIPTextModel.from_pretrained(model_nm, subfolder="text_encoder", use_auth_token=True)
    vae = AutoencoderKL.from_pretrained(model_nm, subfolder="vae", use_auth_token=True)
    unet = UNet2DConditionModel.from_pretrained(model_nm, subfolder="unet", use_auth_token=True)
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

    # Freeze all parameters except for the token embeddings in text encoder
    tm = text_encoder.text_model
    for o in (vae, unet, tm.encoder, tm.final_layer_norm, tm.embeddings.position_embedding):
        for p in o.parameters(): p.requires_grad = False

    train_dataset = TextualInversionDataset(
        images=images, tokenizer=tokenizer, size=512, placeholder_token=placeholder_token,
        repeats=100, template=concept['template'], set="train")

    def create_dataloader(bs=1): return DataLoader(train_dataset, batch_size=bs, shuffle=True)

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    
    pipeline = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae, unet=unet, tokenizer=tokenizer,
            scheduler=PNDMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True),
            safety_checker=None,
            feature_extractor=None)

    def training_function(text_encoder, vae, unet, train_batch_size, gradient_accumulation_steps,
                        lr, max_train_steps, scale_lr, concept):
        accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision='fp16')
        train_dataloader = create_dataloader(train_batch_size)
        if scale_lr: lr = (lr * gradient_accumulation_steps * train_batch_size * accelerator.num_processes)
        optimizer = torch.optim.AdamW(text_encoder.get_input_embeddings().parameters(), lr=lr)
        text_encoder, optimizer, train_dataloader = accelerator.prepare(text_encoder, optimizer, train_dataloader)
        vae.to(accelerator.device).eval()
        unet.to(accelerator.device).eval()

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
        progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0

        
        def save_model(epoch):
            with torch.no_grad():
                learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
                learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
                filename = concept['outpath'] / "learned_embeds-{}.bin".format(epoch)
                torch.save(learned_embeds_dict, filename)
                del learned_embeds, learned_embeds_dict
                preview = pipeline([train_dataset.last_prompt],num_inference_steps=50, guidance_scale=7).images[0]
                print("Generating: {}\n".format(train_dataset.last_prompt))
                preview.save( concept['outpath'] / "learned_embeds-{}.png".format(epoch))
                del preview
                with open(f"{filename.parent / filename.stem}.txt", "w") as f: f.write(train_dataset.last_prompt)
                print("checkpoint done")
                

        for epoch in range(num_train_epochs):
            text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(text_encoder):
                    latents = vae.encode(batch["pixel_values"]).latent_dist.sample().detach() * 0.18215
                    noise = torch.randn(latents.shape).to(latents.device)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()

                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()
                    accelerator.backward(loss)

                    # We only want to optimize the concept embeddings
                    grads = text_encoder.get_input_embeddings().weight.grad
                    index_grads_to_zero = torch.arange(len(tokenizer)) != placeholder_token_id
                    grads.data[index_grads_to_zero, :] = grads.data[index_grads_to_zero, :].fill_(0)
                    optimizer.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if (global_step+1) % batch_config['save_every'] == 0: save_model(global_step+1)
                    
                progress_bar.set_postfix(loss=loss.detach().item())
                if global_step >= max_train_steps: break
        
        
            
        learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
        learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
        torch.save(learned_embeds_dict, concept['outpath'] / "learned_embeds.bin")
        print("training done")
        del bsz, noisy_latents, encoder_hidden_states, noise_pred, index_grads_to_zero
        del grads, loss, timesteps, noise, latents, optimizer, train_dataloader, accelerator, learned_embeds, learned_embeds_dict
    
    
    training_function(text_encoder, vae, unet, train_batch_size=1, gradient_accumulation_steps=4, lr=float(batch_config['lr']),
        max_train_steps=batch_config['max_steps'], scale_lr=True, concept=concept)
    
    del tokenizer, token_embeds, tm, text_encoder, vae, unet, noise_scheduler, train_dataset
    del token_ids, initializer_token_id, placeholder_token_id, images

for k in concept_batch:
    train_concept(k)
    torch.cuda.empty_cache()