import onnxruntime
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the ONNX model
def load_onnx_model(model_path):
    session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    return session, input_name

# Preprocess the image
def preprocess_image(image_path, input_size=(1333, 800)):
    import cv2
    import numpy as np

    # Carregar e converter imagem para RGB
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img.shape[:2]
    
    # Redimensionar a imagem
    img_resized = cv2.resize(img, input_size)
    
    # Converter para formato CHW
    img_tensor = img_resized.transpose(2, 0, 1).astype(np.float32)  # HWC -> CHW
    
    # Normalizar para [0, 1]
    img_tensor = img_tensor / 255.0
    
    # Aplicar normalização usando mean e std
    mean = np.array([123.675, 116.28, 103.53]) / 255.0
    std = np.array([58.395, 57.12, 57.375]) / 255.0
    img_tensor = (img_tensor - mean[:, None, None]) / std[:, None, None]
    
    # Adicionar dimensão do batch
    img_tensor = np.expand_dims(img_tensor, axis=0).astype(np.float32)
    
    return img_tensor, img, original_shape

# Postprocess the model output
def postprocess_output(output, original_shape, input_size=(1333, 800), conf_threshold=0.4):
    boxes, labels = output
    boxes = np.array(boxes[0])
    #scores = np.array(scores)
    labels = np.array(labels[0])

    scores = boxes[:, 4]
    boxes = boxes[:, 0:4]

    # Filter results based on confidence threshold
    keep = scores >= conf_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Scale boxes back to the original image size
    scale_y = original_shape[0] / input_size[1]
    scale_x = original_shape[1] / input_size[0]
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    return boxes, scores, labels

# Visualize the results
def visualize_results(img, boxes, scores, labels, output_path='result.jpg'):
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f'{label}: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Save and plot the image
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    #plt.imshow(img)
    #plt.axis('off')
    #plt.show()

# Main inference function
def run_inference(model_path, image_path, input_size=(1333, 800), conf_threshold=0.5):
    session, input_name = load_onnx_model(model_path)
    img_tensor, img_original, original_shape = preprocess_image(image_path, input_size)

    # Run inference
    output = session.run(None, {input_name: img_tensor})

    # Extract outputs
    boxes, scores, labels = postprocess_output(output, original_shape, input_size, conf_threshold)
    visualize_results(img_original, boxes, scores, labels)

# Example usage
model_path = r"E:\Igor\Downloads\gigasistemica_sandbox_igor\giga_detection\models\empirical_attention\faster_end2end.onnx"
image_path = r"E:\Igor\Downloads\det\giga_new\images\ATCLIN2016-2.jpg"
run_inference(model_path, image_path, input_size=(1333, 800), conf_threshold=0.4)
