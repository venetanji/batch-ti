# batch-ti
Batch training script for textual inversion. Keeping it as simple as possible:

:one: Put files in folders

:two: Edit yaml configuration files

:three: Run the script :rocket:

# Preprocessing

## Expected folder structure
The script expects a specific folder structure for your datasets.
You'll need to prepare your datasets folder before you start.
* Crop the images to 1:1 ratio
* Put each set of images in a folder named with the concept label (should be unique)
* [optional] add a .txt file containing the description with the same name as the image
* Group the concepts based on the template you wish to use. The `subject` template is a good start, but you can create as many templates as you like.

## install requirements
Should be as easy as:
```
pip install -r requirements.txt
```

# Configuration
## batch_config.yaml
Configure the script in `batch_config.yaml`. 

```
datasets_path: datasets

# Output path for concepts
base_outpath: concepts

# pretrained model to use
model: runwayml/stable-diffusion-v1-5

# deftault initialization value for concept tokens
# you can specify the init for each concept in the batch in [batch_folder]/inits.yaml
default_init: person
default_description: painting
save_every: 100
max_steps: 3000
lr: 3e-04
```

Specify which batches and templates you want train on:
```
batches:
  -
    name: my-batch
    templates:
      - subject
  - 
    name: another-batch
    templates:
      - style
```

Specify custom templates:

```
templates:
  subject:
    - a photo of a {}
    - a rendering of a {}
    - a cropped photo of the {}

...
``` 

## inits.yaml
For each batch override the init values for each concept:

```
my-style: anime
another-style: sketch
```
# Train

Just run it!
```
python textual_inversion_batch_training.py
```
Enjoy a walk in the park and a good night of sleep. This will take ~1 hour or more for every concept.

## Safety check
**Safety checker is disabled for previews** during training, enabling it caused OOMs on my gpu. Enable safety at your own risk :rofl:

# [Extra] Publish to huggingface
You can publish the concepts on huggingface with:

```
python publish_concepts.py
```

# Acknowledgments
This script is a modified version of https://github.com/fastai/diffusion-nbs.
Publishing script is inspired from sd-concept-library [Official training colab](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_textual_inversion_training.ipynb). You need to join the sd-concept-library organization in order to publish it there.
