from matplotlib import pyplot as plt
import numpy as np
import torch
import seaborn as sns
import os
import cv2 as cv
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

folder = 'split/test'

test_transform = transforms.Compose([
    transforms.Resize((250, 962)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = datasets.ImageFolder(folder, transform=test_transform)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


model = torch.load('model_epoch_4_val_loss_0.264005.pt')

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def save_wrong_right_predictions(data, target, pred, index, folder='wrong'):
    """
    Save the wrong and right predictions to a specified folder with the correct label appended to the name.

    Parameters:
    data (Tensor): The image data that was input to the model.
    target (Tensor): The true labels.
    pred (Tensor): The predicted labels.
    index (int): The index of the current batch.
    folder (str): The folder where to save the wrong predictions.
    """
    if not os.path.exists('wrong'):
        os.makedirs('wrong')
    if not os.path.exists('right'):
        os.makedirs('right')

    # Assuming data is a batch of images and saving them as .png
    for i, (t, p) in enumerate(zip(target, pred)):
        if t != p:
            t = t.cpu().numpy()[0]
            p = p.cpu().numpy()
            folder = 'wrong'
            # You may need to modify the saving mechanism depending on how your images are stored and formatted
            filename = f'{folder}/{index * data.size(0) + i}_pred{p}_true{t}.png'
            # Save image code here, e.g. plt.imsave(filename, data[i].cpu().numpy())
            img = data[i].cpu().numpy().transpose(1, 2, 0)
            # unnormalize
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            # map to 0-255
            img = (img * 255).astype(np.uint8)

            cv.imwrite(filename, img)
                         
            print(f'Saved wrong prediction to {filename}')

        else:
            t = t.cpu().numpy()[0]
            p = p.cpu().numpy()
            folder = 'right'
            # You may need to modify the saving mechanism depending on how your images are stored and formatted
            filename = f'{folder}/{index * data.size(0) + i}_pred{p}_true{t}.png'
            # Save image code here, e.g. plt.imsave(filename, data[i].cpu().numpy())
            img = data[i].cpu().numpy().transpose(1, 2, 0)
            # unnormalize
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            # map to 0-255
            img = (img * 255).astype(np.uint8)

            cv.imwrite(filename, img)
                         
            print(f'Saved right prediction to {filename}')



def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for index, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            is_correct = pred.eq(target.view_as(pred))
            correct += is_correct.sum().item()
            y_true.extend(target.cpu().numpy())
            y_pred.extend(pred.cpu().numpy().flatten())
            
            # Save wrong predictions
            save_wrong_right_predictions(data, target.view_as(pred), pred.view_as(target), index)
    cm = confusion_matrix(y_true, y_pred)
    specificity_list = []
    for i in range(len(cm)):
        true_negatives = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        false_positives = np.sum(cm[:, i]) - cm[i, i]
        specificity_class = true_negatives / (true_negatives + false_positives)
        specificity_list.append(specificity_class)

    # Calculando a média macro da especificidade
    specificity = np.mean(specificity_list)
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    # plot and save confusion matrix

    # Plotando a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')  # Salvando a figura
    plt.show()
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Precision: {:.4f}, Recall/Sensitivity: {:.4f}, F1: {:.4f}, Specifity: {:.4f}\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy, precision, recall, f1, specificity))

    

test(model, device, test_loader)