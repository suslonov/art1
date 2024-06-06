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
results_file = "/media/Data/data/results/metadata_enriched"
matches_file = "/media/Data/data/results/matches"

device = torch.device("cuda:0")

with open(features_file, "rb") as f:
    (all_outputs, all_ids) = pickle.load(f)

with torch.no_grad():
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

    all_matches = []
    for i, row in tqdm(metadata.iterrows()):
        # feature = features[0]
        feature = features[i]
        matches = {"culture": {}, "medium": {}}

        ll = len(features)//2
        all_dists1 = torch.sum(features[0:ll] * feature, dim=1).to(device)
        all_dists2 = torch.sum(features[ll:] * feature, dim=1).to(device)
        all_dists = torch.cat((all_dists1, all_dists2), 0)

        for culture in cultures:
            selected_indicies = indicies[masks["culture"][culture]]
            k = min(10, selected_indicies.shape[0])
            dists, inds = torch.topk(all_dists[selected_indicies], k, sorted=True)
            matches["culture"][culture] = ids[selected_indicies[inds].cpu().numpy()]
        for medium in media:
            selected_indicies = indicies[masks["medium"][medium]]
            k = min(10, selected_indicies.shape[0])
            dists, inds = torch.topk(all_dists[selected_indicies], k, sorted=True)
            matches["medium"][medium] = ids[selected_indicies[inds].cpu().numpy()]
        all_matches.append(matches)

# tools section

lll = 10000
i = 0
while i <= len(all_matches):
    df_all_matches = pd.DataFrame(all_matches[i:i+lll])
    df_all_matches.to_json(matches_file + ".json" + str(i) + ".json")
    i += lll

print("json 1")

metadata["matches"] = all_matches

i = 0
while i <= len(all_matches):
    metadata.to_json(results_file + str(i) + ".json")
    i += lll

print("here")



lll = 10000
df_all_matches = None
for i in range(48):
    print(i)
    if df_all_matches is None:
        df_all_matches = pd.read_json(matches_file + ".json" + str(i*lll) + ".json")
    else:
        df_all_matches1 = pd.read_json(matches_file + ".json" + str(i*lll) + ".json")
        df_all_matches = pd.concat([df_all_matches, df_all_matches1], axis = 0)


lll = 10000
df_all_matches = None
for i in range(48):
    print(i)
    if df_all_matches is None:
        df_all_matches = pd.read_json(matches_file + ".json" + str(i*lll) + ".json")
        # lc = list(df_all_matches['culture'])
        # lm = list(df_all_matches['medium'])
    else:
        df_all_matches1 = pd.read_json(matches_file + ".json" + str(i*lll) + ".json")
        # lc = lc + list(df_all_matches1['culture'])
        # lm = lm + list(df_all_matches1['medium'])

