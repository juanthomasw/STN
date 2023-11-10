#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.getcwd())
from PIL import Image, ImageDraw, ImageFont
from model.STN import STNet
import numpy as np
import argparse
import torch
import time
import cv2

def convert_image(inp):
    # convert a Tensor to numpy image
    inp = inp.squeeze(0).cpu()
    inp = inp.detach().numpy().transpose((1,2,0))
    inp = 127.5 + inp/0.0078125
    inp = inp.astype('uint8') 

    return inp  

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='STN Demo')
    parser.add_argument("-image", help='image path', default='data/ccpd_weather/吉BTW976.jpg', type=str)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    STN = STNet()
    STN.to(device)
    STN.load_state_dict(torch.load('weights/Final_STN_model.pth', map_location=lambda storage, loc: storage))
    STN.eval()
    
    print("Successful to build STN!")
    
    since = time.time()
    image = cv2.imread(args.image)
    im = cv2.resize(image, (94, 24), interpolation=cv2.INTER_CUBIC)
    im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5)*0.0078125
    data = torch.from_numpy(im).float().unsqueeze(0).to(device)  # torch.Size([1, 3, 24, 94]) 
    transfer = STN(data)

    print("model inference in {:2.3f} seconds".format(time.time() - since))
    
    transformed_img = convert_image(transfer)
    cv2.imshow('transformed', transformed_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
