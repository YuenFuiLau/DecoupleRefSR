from json import encoder
import torch
import torch.nn as nn
import __init_paths
from face_model.gpen_model import Encoder, FullGenerator
import os
import sys

sys.path.append(os.getcwd())

ckpts = torch.load("./weights/GPEN-BFR-512.pth")

model = Encoder(512,512)
model_dict = model.state_dict()

pretrained_dict = {k: v for k, v in ckpts.items() if k in model_dict}
torch.save(pretrained_dict,"./weights/encoder.pth")

model.load_state_dict(pretrained_dict)


ckpts = torch.load("./weights/GPEN-BFR-512.pth")
model = FullGenerator(512,512,8)
model.load_state_dict(ckpts)

last_layer = model.final_linear
torch.save(last_layer.state_dict(),"./weights/proj_linear.pth")
