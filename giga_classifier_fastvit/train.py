import timm
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

folder_train = '/mnt/data/giga_classifier/split/train'
folder_test = '/mnt/data/giga_classifier/split/test'

# load model fastvit_t12 for 2 classes
model = timm.create_model('fastvit_t12', pretrained=True, num_classes=2)

# create augmentation list
train_transform = transforms.Compose([
    transforms.Resize((250, 962)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((250, 962)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# create dataloaders
train_dataset = datasets.ImageFolder(folder_train, transform=train_transform)
test_dataset = datasets.ImageFolder(folder_test, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# train model with eval
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

best_val_loss = float('inf')  # Initialize best validation loss to a very high value

def test_and_evaluate(model, device, test_loader, epoch):
    global best_val_loss
    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.view_as(target).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * accuracy, precision, recall, f1))

    # Save the model if test loss has decreased
    if test_loss < best_val_loss:
        print(f'Validation loss decreased ({best_val_loss:.6f} --> {test_loss:.6f}).  Saving model ...')
        torch.save(model, f'model_epoch_{epoch}_val_loss_{test_loss:.6f}.pt')
        best_val_loss = test_loss

for epoch in range(1, 21):  # 20 epochs
    train(model, device, train_loader, optimizer, epoch)
    test_and_evaluate(model, device, test_loader, epoch)
    scheduler.step()



