import onnxruntime as ort
import numpy as np
from PIL import Image, ImageOps

# Charger le modèle ONNX
session = ort.InferenceSession("mnist_model.onnx")

# Fonction pour prédire une image
def predict_with_onnx(image_path):
    image = Image.open(image_path).convert("L")  # Grayscale
    image = ImageOps.invert(image).resize((28, 28))  # Inverser les couleurs (noir sur blanc)
    image = np.array(image, dtype=np.float32) / 255.0  # Normalisation
    image = image[np.newaxis, np.newaxis, :, :]  # Ajouter batch et channel dims

    # Prédiction avec ONNX
    inputs = {session.get_inputs()[0].name: image}
    outputs = session.run(None, inputs)
    predicted_class = np.argmax(outputs[0])
    print(f"Prédiction : {predicted_class}")

# Tester avec une image
predict_with_onnx("input_images/test_digit.png")
