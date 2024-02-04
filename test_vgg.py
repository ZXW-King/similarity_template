import torchvision.models as models
from torchsummary import summary
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model19 = models.vgg19().to(device)
summary(model19, (3, 224, 224))