import torch
from retinanet.dataloader import CocoDataset, Normalizer, Augmenter, Resizer, collater
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from efficientnet.model import EfficientNet
from retinanet.model import RetinaNet
from retinanet.anchors import Anchors
from retinanet.eval import evaluate
import numpy as np
import collections
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm
#from tqdm.autonotebook import tqdm


###################### DataLoader ##################################

root_dir = "D:\\datasets\\CURE-TSD-FULL"

mean = [0.4379, 0.4951, 0.4512]
std = [0.2384, 0.2503, 0.2485]

training_params = {'batch_size': 1,
                    'shuffle': True,
                    'drop_last': True,
                    'collate_fn': collater,
                    'num_workers': 0}

val_params = {'batch_size': 1,
                'shuffle': False,
                'drop_last': True,
                'collate_fn': collater,
                'num_workers': 0}

training_set = CocoDataset(root_dir=root_dir, set='train',
                               transform=transforms.Compose([Normalizer(mean=mean, std=std),
                                                             Resizer(896)]))

training_generator = DataLoader(training_set, **training_params)

val_set = CocoDataset(root_dir=root_dir, set='val',
                               transform=transforms.Compose([Normalizer(mean=mean, std=std),
                                                             Resizer(896)]))

#val_generator = DataLoader(val_set, **val_params)

#val_generator.num_classes = val_set.num_classes

####################### Model ########################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#pret_path = "models/efn64.pt"

backbone = EfficientNet.from_pretrained('efficientnet-b4')
backbone.source_layer_indexes = [21, 29]

#anchors = Anchors(
#    strides = [2, 4, 8, 16, 32],
#    sizes = [8, 16, 32, 64, 128],
#    ratios = np.array([0.8, 1, 1.2])
#)

anchors = Anchors()

#anchors.strides = [2, 4, 8, 16, 32]
#anchors.sizes = [8, 16, 32, 64, 128]
anchors.ratios = np.array([0.8, 1, 1.2])
anchors.scales = np.array([0.375, 0.6875, 1.3125])

model = RetinaNet(num_classes = 8, backbone_network = backbone, fpn_sizes = [160, 272, 1792], anchors = anchors)
model.to(device)

########################## Training Loop ###################################

writer = SummaryWriter('logs/')

epochs = 10

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

model.train()
loss_hist = collections.deque(maxlen=500)

step = 0

for epoch_num in range(epochs):
    model.train()
    model.freeze_bn()

    epoch_loss = []

    progress_bar = tqdm(training_generator)

    for iter_num, data in enumerate(progress_bar):
        #try:
        optimizer.zero_grad()
        classification_loss, regression_loss = model(
            [data['img'].to(device).float(), data['annot'].to(device).float()])
        classification_loss = classification_loss.mean()
        loss = classification_loss + regression_loss
        if bool(loss == 0):
            continue
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.paramters(), 0.1)
        optimizer.step()
        loss_hist.append(float(loss))
        epoch_loss.append(float(loss))
        progress_bar.set_description('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num,
                                                                            float(classification_loss), float(regression_loss), np.mean(loss_hist)))

        writer.add_scalars('Classification Loss', {'train': classification_loss}, step)
        writer.add_scalars('Regression Loss', {'train': regression_loss}, step)
        writer.add_scalars('Loss', {'train': loss}, step)

        step += 1
        #progress_bar.update()

        #del classification_loss
        #del regression_loss

        #if(iter_num == 10):
        #    print('1 iterations complete, breaking loop ..')
        #    break

        #except Exception as e:
        #    print(e)
        #    #continue
    
    # Apparently, the model evaluation is not working
    model.eval()
    print('Evaluating Model:')
    _, MAP = evaluate(val_set, model)
    print('MAP Score: ',MAP)
    scheduler.step(np.mean(epoch_loss))

    # save the model
    #torch.save(model, os.path.join('weights/', '{}_retinanet_{}_map.pt'.format("EfficientNetb4", epoch_num)))