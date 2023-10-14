import torch
from byol_pytorch import BYOL
import torchvision
from torchvision import models
import numpy as np
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from torchvision.models.vision_transformer import VisionTransformer


resnet = models.resnet50(pretrained=True).cuda()
# self.vit_encoder = VisionTransformer(
#     image_size=224,
#     patch_size=16,
#     num_layers=12,
#     num_heads=6,
#     hidden_dim=384,
#     mlp_dim=1536,
#     representation_size=512,
# )

learner = BYOL(
    resnet,
    image_size = 224,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
im_mean = np.asarray([0.485, 0.456, 0.406])
im_std = np.asarray([0.229, 0.224, 0.225])
load_batch_size = 32

train_dataset = torchvision.datasets.ImageFolder("datasets/grabber", transform=Compose(
[
    ToTensor(),
    Normalize(im_mean, im_std),
]))
dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=16)


for epoch in range(100):
    print(epoch)
    for images, label in tqdm(iter(dataloader)):
        loss = learner(images.cuda())
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder

# save your improved network
torch.save(resnet.state_dict(), './improved-net.pt')