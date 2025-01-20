import numpy as np
import onnxruntime
import torch
import os
import cv2 as cv

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

def load_classifier_model(path='giga_classifier_fastvit/models/model_epoch_4_val_loss_0.264005.pt'):
    model = torch.load(path)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model

def predict_classifier_model(image_path, model, device, class_names):
    """
    Realiza a predição em uma única imagem usando o modelo PyTorch fornecido.

    Parameters:
    - image_path (str): Caminho para a imagem.
    - model (torch.nn.Module): Modelo PyTorch já carregado.
    - device (torch.device): Dispositivo para executar o modelo (CPU ou GPU).
    - class_names (list): Lista de nomes das classes.

    Returns:
    - predicted_class (str): Classe prevista para a imagem.
    """
    # Transformações de pré-processamento
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensiona para o tamanho esperado pelo modelo
        transforms.ToTensor(),  # Converte para tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliza com mean/std
    ])
    
    # Carregar e transformar a imagem
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Adiciona dimensão do batch

    # Envia para o dispositivo correto
    image = image.to(device)

    # Modo de avaliação
    model.eval()
    with torch.no_grad():
        # Realiza a predição
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # Classe com maior probabilidade
        predicted_class = class_names[predicted.item()]

    return predicted_class

def load_detection_model(model_path='giga_detection/models/empirical_attention/faster_end2end.onnx'):
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    return session, input_name

def predict_detection_model(model, image_path, device, input_size=(1333, 800), conf_threshold=0.5):
    """
    Executa a inferência em uma imagem usando um modelo PyTorch.

    Parameters:
    - model (torch.nn.Module): Modelo PyTorch já carregado.
    - image_path (str): Caminho para a imagem.
    - device (torch.device): Dispositivo (CPU ou GPU).
    - input_size (tuple): Tamanho para redimensionamento da imagem.
    - conf_threshold (float): Threshold de confiança para filtrar resultados.

    Returns:
    - results (list): Lista de detecções contendo boxes, scores e labels.
    """
    # Pré-processar a imagem
    def preprocess_image(image_path, input_size):
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        original_shape = img.shape[:2]
        img_resized = cv.resize(img, input_size)
        img_tensor = img_resized.transpose(2, 0, 1).astype(np.float32)  # HWC -> CHW
        img_tensor = img_tensor / 255.0
        mean = np.array([123.675, 116.28, 103.53]) / 255.0
        std = np.array([58.395, 57.12, 57.375]) / 255.0
        img_tensor = (img_tensor - mean[:, None, None]) / std[:, None, None]
        img_tensor = np.expand_dims(img_tensor, axis=0).astype(np.float32)
        return img_tensor, img, original_shape

    # Pós-processamento do output do modelo
    def postprocess_output(output, original_shape, input_size, conf_threshold):
        boxes, scores, labels = output
        boxes = boxes * [original_shape[1] / input_size[0], original_shape[0] / input_size[1],
                         original_shape[1] / input_size[0], original_shape[0] / input_size[1]]
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []
        for box, score, label in zip(boxes, scores, labels):
            if score >= conf_threshold:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_labels.append(label)
        return filtered_boxes, filtered_scores, filtered_labels

    # Visualizar resultados
    def visualize_results(img, boxes, scores, labels):
        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = map(int, box)
            cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text = f"{label}: {score:.2f}"
            cv.putText(img, text, (x_min, y_min - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv.imshow("Results", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # Carregar e processar imagem
    img_tensor, img_original, original_shape = preprocess_image(image_path, input_size)
    img_tensor = torch.tensor(img_tensor).to(device)

    # Modelo em modo de avaliação
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)

    # Extração de outputs (ajuste conforme arquitetura do modelo)
    boxes = outputs['boxes'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    labels = outputs['labels'].cpu().numpy()

    # Pós-processar e visualizar resultados
    boxes, scores, labels = postprocess_output((boxes, scores, labels), original_shape, input_size, conf_threshold)
    #visualize_results(img_original, boxes, scores, labels)

    return boxes, scores, labels