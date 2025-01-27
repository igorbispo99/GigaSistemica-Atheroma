import argparse
import json
import numpy as np
import onnxruntime
import torch
import os
import cv2 as cv

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

def load_classifier_model(path='giga_classifier_fastvit/models/model_epoch_4_val_loss_0.264005.pt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(path, map_location=device)

    model.eval()

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

def predict_detection_model(onnx_session, input_name, image_path, input_size=(1333, 800), conf_threshold=0.5):
    """
    Executa a inferência em uma imagem usando um modelo ONNX de detecção.

    Parameters:
    - onnx_session (onnxruntime.InferenceSession): Sessão ONNX carregada.
    - input_name (str): Nome da entrada no modelo ONNX.
    - image_path (str): Caminho para a imagem.
    - input_size (tuple): Tamanho para redimensionamento da imagem (largura, altura).
    - conf_threshold (float): Threshold de confiança para filtrar resultados.

    Returns:
    - boxes (list): Coordenadas filtradas das detecções [x_min, y_min, x_max, y_max].
    - scores (list): Confianças filtradas das detecções.
    - labels (list): Rótulos filtrados das detecções.
    """

    # Função de pré-processamento da imagem
    def preprocess_image(img_path, size):
        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        original_shape = img.shape[:2]
        img_resized = cv.resize(img, size)
        img_tensor = img_resized.transpose(2, 0, 1).astype(np.float32)  # HWC -> CHW
        img_tensor = img_tensor / 255.0

        # Normalização típica (valores exemplo)
        mean = np.array([123.675, 116.28, 103.53]) / 255.0
        std = np.array([58.395, 57.12, 57.375]) / 255.0
        img_tensor = (img_tensor - mean[:, None, None]) / std[:, None, None]

        # Adicionando dimensão de batch
        img_tensor = np.expand_dims(img_tensor, axis=0).astype(np.float32)

        return img_tensor, img, original_shape

    # Função de pós-processamento da saída do modelo
    def postprocess_output(output, original_shape, size, threshold):
        """
        Supondo que o modelo retorne [boxes, scores, labels] ou um formato similar:
         - boxes deve ter shape (N, 4)
         - scores deve ter shape (N,)
         - labels deve ter shape (N,)

        Caso seu modelo retorne algo diferente, ajuste aqui para adequar à sua arquitetura.
        """
        boxes, scores, labels = output

        # Ajusta coordenadas das boxes para o tamanho original da imagem
        h_scale = original_shape[0] / size[1]
        w_scale = original_shape[1] / size[0]
        boxes = boxes * [w_scale, h_scale, w_scale, h_scale]

        filtered_boxes, filtered_scores, filtered_labels = [], [], []
        for box, score, label in zip(boxes, scores, labels):
            if score >= threshold:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_labels.append(label)
        return filtered_boxes, filtered_scores, filtered_labels

    # Pré-processar a imagem
    img_tensor, img_original, original_shape = preprocess_image(image_path, input_size)

    # Inferência no modelo ONNX
    outputs = onnx_session.run(None, {input_name: img_tensor})

    # Aqui assumimos que a saída é uma lista/tupla [boxes, scores, labels].
    # Ajuste os índices caso a saída seja em outra ordem.
    boxes_scores = outputs[0][0]
    labels = outputs[1][0]

    # unpack boxes and scores, boxes_scores shape is like: (N_DETECTIONS, 5)
    # the last element is the class score
    boxes = boxes_scores[:, :-1]
    scores = boxes_scores[:, -1]

    # Pós-processamento para filtrar resultados
    boxes, scores, labels = postprocess_output((boxes, scores, labels), original_shape, input_size, conf_threshold)

    return boxes, scores, labels


def load_segmentation_model(checkpoint_path='giga_segmentation/checkpoint.pth'):
    from giga_segmentation import DC_UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DC_UNet.DC_Unet(1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict_segmentation_model(image_path, model, device, test_size=352, save_mask=False):
    image = Image.open(image_path).convert('L')
    width, height = image.size
    num_rows, num_cols = 2, 3
    cell_width = width // num_cols
    cell_height = height // num_rows
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    cells_to_process = [(1, 0), (1, 2)]

    for row_idx, col_idx in cells_to_process:
        left = col_idx * cell_width
        upper = row_idx * cell_height
        right = width if col_idx == num_cols - 1 else (col_idx + 1) * cell_width
        lower = height if row_idx == num_rows - 1 else (row_idx + 1) * cell_height
        cell_image = image.crop((left, upper, right, lower))
        cell_image_resized = cell_image.resize((test_size, test_size))
        cell_tensor = transforms.ToTensor()(cell_image_resized)
        cell_tensor = transforms.Normalize([0.5], [0.5])(cell_tensor)
        cell_tensor = cell_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(cell_tensor)

        prediction = prediction.sigmoid().cpu().numpy().squeeze()
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
        prediction = (prediction >= 0.5).astype(np.uint8)
        prediction_resized = Image.fromarray(prediction * 255).resize((right - left, lower - upper), resample=Image.NEAREST)
        combined_mask[upper:lower, left:right] = np.array(prediction_resized) // 255

    if save_mask:
        out_mask = Image.fromarray((combined_mask * 255).astype(np.uint8))
        out_mask.save("segmentation_output.png")
    return combined_mask

def predict_pipeline(
    image_path,
    classifier_model,
    detection_model,
    segmentation_model,
    device,
    class_names,
    detection_conf_threshold=0.5,
    test_size=352,
    output_png="output.png",
    output_json="output.json"
):
    # 1) Classificação
    predicted_class = predict_classifier_model(image_path, classifier_model, device, class_names)
    results = {"classification": predicted_class}

    # Carregar imagem para visualização final
    original_img = cv.imread(image_path)
    if original_img is None:
        raise ValueError(f"Não foi possível ler a imagem em {image_path}")
    h, w = original_img.shape[:2]

    # 2) Se a classificação indicar presença de ateroma:
    if predicted_class.lower().find("nao") == -1:  # Exemplo: caso não contenha "não" no nome da classe
        # 2a) Detecção
        boxes, scores, labels = predict_detection_model(
            detection_model,
            image_path,
            device,
            input_size=(1333, 800),
            conf_threshold=detection_conf_threshold
        )
        # Armazenar resultados de detecção
        detection_info = []
        for box, score, label in zip(boxes, scores, labels):
            detection_info.append({
                "box": [float(x) for x in box],
                "score": float(score),
                "label": int(label)
            })
            x_min, y_min, x_max, y_max = map(int, box)
            cv.rectangle(original_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv.putText(original_img, f"{label}:{score:.2f}", (x_min, max(0, y_min - 10)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        results["detections"] = detection_info

        # 2b) Segmentação
        combined_mask = predict_segmentation_model(
            image_path, segmentation_model, device, test_size=test_size, save_mask=False
        )
        # Desenhar máscara em vermelho (tom sobre a imagem)
        # Converter para RGBA temporariamente se quiser alpha, mas aqui faremos direto em BGR
        mask_indices = np.where(combined_mask > 0)
        overlay_color = (0, 0, 255)
        original_img[mask_indices] = overlay_color

    # 3) Salvar imagem resultante com caixas/máscaras
    cv.imwrite(output_png, original_img)

    # 4) Salvar JSON com informações
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Caminho para a imagem ou pasta de imagens.")
    parser.add_argument("--output_path", default="results", help="Caminho para salvar os resultados.")
    parser.add_argument("--proceed_when_negative", action="store_true", help="Prosseguir com detecção/segmentação mesmo se classificador for negativo.")
    parser.add_argument("--classifier_model_path", default="giga_classifier_fastvit/models/model_epoch_4_val_loss_0.264005.pt",
                        help="Caminho para o modelo de classificação.")
    parser.add_argument("--detection_model_path", default="giga_detection/models/empirical_attention/faster_end2end.onnx",
                        help="Caminho para o modelo de detecção.")
    parser.add_argument("--segmentation_model_path", default="giga_segmentation/checkpoint.pth",
                        help="Caminho para o modelo de segmentação.")
    parser.add_argument("--class_names", nargs="+", default=["Nao_Ateroma", "Ateroma"], help="Lista de nomes de classe.")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier_model = load_classifier_model(args.classifier_model_path)
    detection_model, detection_input = load_detection_model(args.detection_model_path)
    segmentation_model = load_segmentation_model(args.segmentation_model_path)
    segmentation_model.eval().to(device)

    if os.path.isfile(args.input_path):
        image_list = [args.input_path]
    else:
        exts = [".jpg", ".jpeg", ".png"]
        image_list = [
            os.path.join(args.input_path, f)
            for f in os.listdir(args.input_path)
            if os.path.splitext(f)[1].lower() in exts
        ]

    for img_path in image_list:
        predicted_class = predict_classifier_model(img_path, classifier_model, device, args.class_names)
        results = {"classification": predicted_class}
        img = cv.imread(img_path)
        if img is None:
            continue

        out_png = os.path.join(args.output_path, os.path.splitext(os.path.basename(img_path))[0] + "_output.png")
        out_json = os.path.join(args.output_path, os.path.splitext(os.path.basename(img_path))[0] + "_output.json")

        do_detection_seg = True
        if "nao" in predicted_class.lower() and not args.proceed_when_negative:
            do_detection_seg = False

        if do_detection_seg:
            boxes, scores, labels = predict_detection_model(detection_model, detection_input, img_path, conf_threshold=0.5)
            det_info = []
            for box, score, label in zip(boxes, scores, labels):
                det_info.append({
                    "box": [float(x) for x in box],
                    "score": float(score),
                    "label": int(label)
                })
                x_min, y_min, x_max, y_max = map(int, box)
                cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv.putText(img, f"{label}:{score:.2f}", (x_min, max(0, y_min - 10)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            results["detections"] = det_info

            combined_mask = predict_segmentation_model(img_path, segmentation_model, device)
            mask_indices = np.where(combined_mask > 0)
            img[mask_indices] = (0, 0, 255)

        cv.imwrite(out_png, img)

        with open(out_json, "w") as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()    