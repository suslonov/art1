import os

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import pickle
import shutil

data_dir = '/media/Data/data/images'
metadata_fn = "/media/Data/data/metadata.json"
features_dir = "/media/Data/data/features"
features_file = os.path.join(features_dir, "pytorch_rn50.pkl")
featurize_images = True
device = torch.device("cuda:0")

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model = models.resnet50(pretrained=True)
model.eval()
model.to(device)
cut_model = nn.Sequential(*list(model.children())[:-1])

all_outputs = []
all_ids = []

with open(features_file, "rb") as f:
    (all_outputs, all_ids) = pickle.load(f)

metadata = pd.read_json(metadata_fn, lines=True)



input_file = "/home/anton/git/art/media/P1160781.JPG"

with open(input_file, "rb") as f:
    image = Image.open(f).convert("RGB")
    inputs = data_transform(image)

inputs = inputs.reshape([1, 3, 224, 224])

inputs = inputs.to(device)
outputs = torch.squeeze(cut_model(inputs)).detach()

k = 10
with torch.no_grad():
    features = torch.from_numpy(all_outputs).float().to("cpu:0")
    features = features / torch.sqrt(torch.sum(features ** 2, dim=1, keepdim=True))
    features = features.to(device)
    # indicies = torch.arange(0, features.shape[0]).to(device)
    print("loaded features")

with torch.no_grad():
    ll = len(features)//2
    all_dists1 = torch.sum(features[0:ll] * outputs, dim=1).to(device)
    all_dists2 = torch.sum(features[ll:] * outputs, dim=1).to(device)
    all_dists = torch.cat((all_dists1, all_dists2), 0)

    # all_dists = torch.sum(features * outputs, dim=1).to(device)
    dists, inds = torch.topk(all_dists, k, sorted=True)
    matches = [inds.cpu().numpy()]

image_matches = []
for m in matches[0]:
    with open(os.path.join(data_dir, metadata.iloc[m].id + ".jpg"), "rb") as f:
        shutil.copyfile(os.path.join(data_dir, metadata.iloc[m].id + ".jpg"), os.path.join("/home/anton/git/art/example2", metadata.iloc[m].id + ".jpg"))
        image_matches.append(Image.open(f).convert("RGB"))

