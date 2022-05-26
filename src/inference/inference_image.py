import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_face
import warnings
warnings.filterwarnings('ignore')

def main(args):

    model=Detector()
    model=model.to(device)
    cnn_sd=torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    frame = cv2.imread(args.input_image)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_detector = get_model("resnet50_2020-07-20", max_size=max(frame.shape),device=device)
    face_detector.eval()

    face_list=extract_face(frame,face_detector)

    with torch.no_grad():
        img=torch.tensor(face_list).to(device).float()/255
        # torchvision.utils.save_image(img, f'test.png', nrow=8, normalize=False, range=(0, 1))
        pred=model(img).softmax(1)[:,1].cpu().data.numpy().tolist()

    print(f'fakeness: {max(pred):.4f}')


if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-i',dest='input_image',type=str)
    args=parser.parse_args()

    main(args)

