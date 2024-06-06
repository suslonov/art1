import os
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import pickle

data_dir = '/media/Data/data/images'
metadata_fn = "/media/Data/data/metadata.json"
features_dir = "/media/Data/data/features"
features_file = os.path.join(features_dir, "pytorch_rn50.pkl")
device = torch.device("cuda:0")


f=open(features_file, "rb")
torch.no_grad()

(all_outputs, all_ids) = pickle.load(f)
all_urls = np.array(pd.read_json(metadata_fn, lines=True).loc[:, "Thumbnail_Url"])
features = torch.from_numpy(all_outputs).float().to("cpu:0")
features = features / torch.sqrt(torch.sum(features ** 2, dim=1, keepdim=True))
features = features.to(device)
indicies = torch.arange(0, features.shape[0]).to(device)
print("loaded features")

metadata = pd.read_json(metadata_fn, lines=True)
culture_arr = np.array(metadata["Culture"])
cultures = metadata.groupby("Culture").count()["id"].sort_values(ascending=False).index.to_list()
media_arr = np.array(metadata["Classification"])
media = metadata.groupby("Classification").count()["id"].sort_values(ascending=False).index.to_list()
ids = np.array(metadata["id"])

masks = {"culture": {}, "medium": {}}
for culture in cultures:
    masks["culture"][culture] = torch.from_numpy(culture_arr == culture).to(device)
for medium in media:
    masks["medium"][medium] = torch.from_numpy(media_arr == medium).to(device)

all_matches1 = []
all_matches2 = []
for i, row in tqdm(metadata.iterrows()):
    # feature = features[0]
    feature = features[i]
    matches = {"culture": {}, "medium": {}}

    ll = len(features)//2
    all_dists1 = torch.sum(features[0:ll] * feature, dim=1).to(device)
    for culture in cultures:
        selected_indicies = indicies[0:ll][masks["culture"][culture][0:ll]]
        k = min(10, selected_indicies.shape[0])
        dists, inds = torch.topk(all_dists1[selected_indicies], k, sorted=True)
        # matches["culture"][culture] = ids[selected_indicies[inds].cpu().numpy()]
        matches["culture"][culture] = np.concatenate((dists.cpu().numpy().reshape(10,1), ids[selected_indicies[inds].cpu().numpy()].reshape(10,1)), axis=1)
    for medium in media:
        selected_indicies = indicies[0:ll][masks["medium"][medium][0:ll]]
        k = min(10, selected_indicies.shape[0])
        dists, inds = torch.topk(all_dists1[selected_indicies], k, sorted=True)
        # matches["medium"][medium] = ids[selected_indicies[inds].cpu().numpy()]
        matches["medium"][medium] = np.concatenate((dists.cpu().numpy().reshape(10,1), ids[selected_indicies[inds].cpu().numpy()].reshape(10,1)), axis=1)
    all_matches1.append(matches)
    del all_dists1

    matches = {"culture": {}, "medium": {}}
    all_dists2 = torch.sum(features[ll:] * feature, dim=1).to(device)
    # culture = cultures[0]
    for culture in cultures:
        selected_indicies = indicies[ll:][masks["culture"][culture][ll:]]
        selected_indicies=selected_indicies.add(-ll)
        k = min(10, selected_indicies.shape[0])
        dists, inds = torch.topk(all_dists2[selected_indicies], k, sorted=True)
        # matches["culture"][culture] = ids[selected_indicies[inds].cpu().numpy()]
        matches["culture"][culture] = np.concatenate((dists.cpu().numpy().reshape(10,1), ids[selected_indicies[inds].cpu().numpy()].reshape(10,1)), axis=1)
    for medium in media:
        selected_indicies = indicies[ll:][masks["medium"][medium][ll:]]
        selected_indicies=selected_indicies.add(-ll)
        k = min(10, selected_indicies.shape[0])
        dists, inds = torch.topk(all_dists2[selected_indicies], k, sorted=True)
        # matches["medium"][medium] = ids[selected_indicies[inds].cpu().numpy()]
        matches["medium"][medium] = np.concatenate((dists.cpu().numpy().reshape(10,1), ids[selected_indicies[inds].cpu().numpy()].reshape(10,1)), axis=1)
    all_matches2.append(matches)


# tools section

# metadata["matches"] = all_matches

metadata.to_json("results/metadata_enriched.json")
print("here")


