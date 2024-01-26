import time
from torch.utils.data import Dataset
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.notebook import tqdm
import torchvision.models as models


transform = transforms.Compose([
    transforms.ToPILImage(),  
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5], std=[0.5])  
])


def read_image(index):
    path = os.path.join("dataset/Extracted Faces", index[0], index[1])
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def train_test_split(directory, split=0.8):
    all_items = os.listdir(directory)
    folders = [item for item in all_items if os.path.isdir(os.path.join(directory, item))]
    num_train = int(len(folders)*split)
    
    random.shuffle(folders)
    
    train_list, test_list = {}, {}
    
    # Creating Train-list
    for folder in folders[:num_train]:
        num_files = len(os.listdir(os.path.join(directory, folder)))
        train_list[folder] = num_files
    
    # Creating Test-list
    for folder in folders[num_train:]:
        num_files = len(os.listdir(os.path.join(directory, folder)))
        test_list[folder] = num_files  
    
    return train_list, test_list


def generate_triplet(directory, folder_list, max_files=10):
    triplets = []
    folders = list(folder_list.keys())
    
    for folder in folders:
        path = os.path.join(directory, folder)
        files = list(os.listdir(path))[:max_files]
        num_files = len(files)
        
        for i in range(num_files-1):
            for j in range(i+1, num_files):
                anchor = (folder, f"{i}.jpg")
                positive = (folder, f"{j}.jpg")

                neg_folder = folder
                while neg_folder == folder:
                    neg_folder = random.choice(folders)
                neg_file = random.randint(0, folder_list[neg_folder]-1)
                negative = (neg_folder, f"{neg_file}.jpg")

                triplets.append((anchor, positive, negative))
            
    random.shuffle(triplets)
    return triplets


def transform_triplet(triplet_list, batch_size=256, apply_transform=True):
    batch_steps = len(triplet_list) // batch_size
    
    for i in range(batch_steps + 1):
        anchor, positive, negative = [], [], []

        j = i * batch_size
        while j < (i + 1) * batch_size and j < len(triplet_list):
            a, p, n = triplet_list[j]
            a_img = read_image(a)
            p_img = read_image(p)
            n_img = read_image(n)

            if a_img is not None and p_img is not None and n_img is not None:
                if apply_transform:
                    a_img = transform(a_img)
                    p_img = transform(p_img)
                    n_img = transform(n_img)

                anchor.append(a_img)
                positive.append(p_img)
                negative.append(n_img)
            j += 1

        if not anchor or not positive or not negative:
            continue

        anchor = torch.stack(anchor)
        positive = torch.stack(positive)
        negative = torch.stack(negative)

        yield ([anchor, positive, negative])



def accuracy_model(siamese_model, test_triplet, batch_size=256):
    siamese_model.eval()
    pos_scores, neg_scores = [], []
    
    with torch.no_grad():
        for data in transform_triplet(test_triplet, batch_size=batch_size):
            anchor, positive, negative = data
            ap_distance, an_distance = siamese_model(anchor, positive, negative)
            pos_scores.extend(ap_distance.cpu().numpy())
            neg_scores.extend(an_distance.cpu().numpy())

    accuracy = np.sum(np.array(pos_scores) < np.array(neg_scores)) / len(pos_scores)
    ap_mean, an_mean = np.mean(pos_scores), np.mean(neg_scores)
    ap_stds, an_stds = np.std(pos_scores), np.std(neg_scores)
    
    print(f"Accuracy on test = {accuracy:.5f}")
    return accuracy, ap_mean, an_mean, ap_stds, an_stds