import os
from pydoc import cli
import re
from typing import List, Optional
import click
import dnnlib
import legacy

from xml.etree.ElementTree import PI
import torch
torch.manual_seed(1234)
import torch.nn as nn
import clip
from PIL import Image
import numpy as np
import cv2

torch.manual_seed(1500)

# debug
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def save_to(img, filename):
    # img: nchw
    image = torch.clone(img[0, :, :, :]).detach()
    unNormalize = UnNormalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    image = unNormalize(image)
    image = np.transpose(image.cpu().numpy(), (1, 2, 0)) * 255
    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device=device)

# text = clip.tokenize(["old man"]).to(device) # great
# text = clip.tokenize(["happy young man"]).to(device) # great
# text = clip.tokenize(["young girl"]).to(device) # ok
text = clip.tokenize(["angry man"]).to(device) # maybe

c = 10
mse = nn.MSELoss(reduction="sum")

# load stylegan
network_pkl = "./pretrained/ffhq.pkl"
with dnnlib.util.open_url(network_pkl) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

# Labels.
label = torch.zeros([1, G.c_dim], device=device)
print('label={}'.format(label))

# Gan config
truncation_psi = 1.0
noise_mode = "const"

z = (torch.randn((1, G.z_dim), device=device)).requires_grad_() # z is latent
# z = torch.from_numpy(np.random.RandomState(1000).randn(1, G.z_dim)).to(device).requires_grad_()

print("z.requires_grad={}".format(z.requires_grad))
optimizer = torch.optim.Adam([z], lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.75)

# debug
def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BILINEAR),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def normalize_image(img):
    # img: nchw
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                       device=device).reshape(1, 3, 1, 1)
    mean.expand(1, 3, img.shape[2], img.shape[3])
    std.expand(1, 3, img.shape[2], img.shape[3])
    return (img - mean) / std

def custom_transform(image, size):
    image = torch.nn.functional.interpolate(image, (size, size), mode='bilinear', align_corners=False)
    image = normalize_image(image)
    return image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
    print("from torch")
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR
    print("from image")

# clip_input_size = model.input_resolution.item()
clip_input_size = model.visual.input_resolution
print("clip_input_size={}".format(clip_input_size))

epochs = 1000
my_pre = _transform(clip_input_size)
for i in range(epochs):
    optimizer.zero_grad()
    gan_img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    gan_img = (gan_img * 127.5 + 128).clamp(0, 255) / 255.0 # nchw
    # print("gan_img.shape={}".format(gan_img.shape))
    # clip_img = preprocess(gan_img).unsqueeze(0).to(device)
    clip_img = custom_transform(gan_img, clip_input_size).to(device)
    if 0 == i:
        save_to(clip_img, 'clipin_{}.png'.format("original"))

    image_features = model.encode_image(clip_img)
    text_features = model.encode_text(text)

    # print("image_features.shape={}".format(image_features.shape))
    # print("text_features.shape={}".format(text_features.shape))
    loss =  1.0 - torch.nn.functional.cosine_similarity(image_features, text_features)
    print("epoch={}, loss={}".format(i, loss.item()))
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        save_to(clip_img, 'clipin_{}.png'.format(i))
    scheduler.step() # Don't use if loss doesn't drop?

save_to(clip_img, 'clipin_{}.png'.format(epochs))
# Image.fromarray(gan_img[0].cpu().numpy(), 'RGB').save('ganclip_{}.png'.format(epochs))