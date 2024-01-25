import torch
import torch.nn.functional as F
import torch.nn as nn
from model import Generator

ckpt_path = "stargan_new_6_leaky/models/330000-model/330000-G.ckpt"
saved_checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
G = Generator(64, 6, 6)
G.to('cpu')
G.load_state_dict(saved_checkpoint, strict = False)
trace_model = torch.jit.trace(G, example_input_tensor)
trace_model.save('stargan_330000.pt')