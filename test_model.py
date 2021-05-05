import torch
from efficientnet.model import EfficientNet
from retinanet.model import RetinaNet
from retinanet.anchors import Anchors
import numpy as np

device = torch.device('cuda')

inputs = torch.rand(1, 3, 896, 896)
inputs = inputs.to(device)

backbone = EfficientNet.from_pretrained('efficientnet-b4')
backbone.source_layer_indexes = [21, 29]

anchors = Anchors()
anchors.ratios = np.array([0.8, 1, 1.2])
anchors.scales = np.array([0.375, 0.6875, 1.3125])

pret_path = "models/efn64.pt"

backbone = EfficientNet.from_pretrained_model('efficientnet-b4', pret_path)
backbone.source_layer_indexes = [21, 29]

model = RetinaNet(num_classes = 8, backbone_network = backbone, fpn_sizes = [160, 272, 1792], anchors = anchors)
model.to(device)
model.eval()

output = model(inputs)