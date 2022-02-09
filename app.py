# Importing libraries for gradio app
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as tt
from PIL import Image



# Moving data to CPU
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)



# Defining our Class for just prediction
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}



# Defining our finetuned Resnet50 Architecture with our Classification layer
class IndianFoodModelResnet50(ImageClassificationBase):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.network = models.resnet50(pretrained=pretrained)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)



# Prediction method
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)



# Initialising our model and moving it to CPU
classes = ['burger',  'butter_naan',  'chai',  'chapati',  'chole_bhature', 
           'dal_makhani',  'dhokla',  'fried_rice',  'idli',  'jalebi',  
           'kaathi_rolls',  'kadai_paneer',  'kulfi',  'masala_dosa',  'momos',
           'paani_puri',  'pakode',  'pav_bhaji',  'pizza',  'samosa']
model = IndianFoodModelResnet50(len(classes), pretrained=True)
device = 'cpu'
to_device(model, device);


# Loading the model
ckp_path = 'indianFood-resnet50.pth'
model.load_state_dict(torch.load(ckp_path, map_location=torch.device('cpu')))
model.eval()



# Image preprocessing before prediction
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
img_tfms = tt.Compose([tt.Resize((224, 224)),
                        tt.ToTensor(), 
                        tt.Normalize(*stats, inplace = True)])

def predict_image(image, model):
    xb = to_device(image.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return classes[preds[0].item()]



# Function handling input, processing and output
def classify_image(path):
    img = Image.open(path)
    img = img_tfms(img)
    label = predict_image(img, model)
    return label



# Defining gradio interface functions
image = gr.inputs.Image(shape=(224, 224), type="filepath")
label = gr.outputs.Label(num_top_classes=1)

article = "<p style='text-align: center'><a href='https://' target='_blank'>DesiVisionNet</a> | <a href='https://github.com/kunal-bhadra/DesiVisionNet' target='_blank'>GitHub Repo</a></p>"


gr.Interface(
    fn=classify_image, 
    inputs=image, 
    outputs=label, 
    examples = [["idli.jpg"], ["naan.jpg"]],
    theme = "huggingface",
    title = "DesiVisionNet: Indian Food Vision with ResNet",
    description = "This is a Gradio demo for multi-class image classification of Indian food amongst 20 classes. The DesiVisionNet achieved 90% accuracy on our test dataset, performing well for a relatively efficient model. See the GitHub project page for detailed information below. Here, we provide a demo for real-world food classification. To use it, simply upload your image, or click one of the examples to load them.",
    article = article
).launch()