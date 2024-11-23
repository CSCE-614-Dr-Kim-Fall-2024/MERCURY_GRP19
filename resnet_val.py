import torch
from torch.utils.data import DataLoader
from torchvision import models, datasets
from torchvision.models import ResNet101_Weights
from tqdm import tqdm

ofile = open("out/resnet_run.log","w")
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)
ofile.write("Using Device: " + str(device)+"\n")

# Load ResNet101 with pretrained weights
weights = ResNet101_Weights.IMAGENET1K_V2
model = models.resnet101(weights=weights)
model = model.to(device)
model.eval()
print("Created Model Resnet101")

# Define ImageNet test dataset path
imagenet_test_dir = "imagenet_set"

# Get preprocessing transforms directly from the weights
transform = weights.transforms()


# Load ImageNet test dataset
test_dataset = datasets.ImageNet(imagenet_test_dir, split='val', transform=transform)
select = torch.randint(0, len(test_dataset), (100*8,)).tolist()
ofile.write("Selected Images: "+ str(select)+"\n")

test_dataset = torch.utils.data.Subset(test_dataset, select)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=4)
print("Test loader ready")
ofile.write("Length of test loader: "+ str(len(test_loader))+"\n")

# Function to evaluate accuracy
def evaluate_model(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Evaluate the model
accuracy = evaluate_model(model, test_loader)
print(f"Model accuracy on ImageNet test dataset: {accuracy * 100:.2f}%")
ofile.write(f"Model accuracy on ImageNet test dataset: {accuracy * 100:.2f}%")
ofile.flush()
ofile.close()
