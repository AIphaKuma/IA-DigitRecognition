import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
from models.cnn import CNN
from models.autoencoder import AutoEncoder

# Charger les modèles
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Charger le modèle CNN pour la classification
cnn_model = CNN().to(device)
cnn_model.load_state_dict(torch.load('./best_mnist_model.pth', map_location=device))
cnn_model.eval()

# Charger l'AutoEncoder pour la reconstruction
autoencoder = AutoEncoder().to(device)
autoencoder.load_state_dict(torch.load('./autoencoder.pth', map_location=device))
autoencoder.eval()

def preprocess_image(image):
    """
    Prétraitement de l'image (remplissage, redimensionnement, conversion en tenseur).
    """
    # Étape 1 : Conversion en niveaux de gris
    image = image.convert("L")
    st.write("### Étape 1 : Conversion en niveaux de gris")
    st.image(image, caption="Image en niveaux de gris", use_column_width=True)

    # Étape 2 : Amélioration du contraste
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(15)
    st.write("### Étape 2 : Amélioration du contraste")
    st.image(image, caption="Image après amélioration du contraste", use_column_width=True)

    # Étape 3 : Binarisation (seuil)
    image = image.point(lambda x: 255 if x > 128 else 0, mode='1')
    st.write("### Étape 3 : Binarisation")
    st.image(image, caption="Image après binarisation", use_column_width=True)

    # Étape 4 : Remplissage intérieur
    image_np = np.array(image, dtype=np.uint8) * 255
    h, w = image_np.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(image_np, mask, (0, 0), 255)
    image_np = cv2.bitwise_not(image_np)
    st.write("### Étape 4 : Remplissage intérieur du chiffre")
    st.image(image_np, caption="Image après remplissage intérieur", use_column_width=True)

    # Étape 5 : Redimensionnement et normalisation
    image = Image.fromarray(image_np)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor_image = transform(image).unsqueeze(0).to(device)
    st.write("### Étape 5 : Transformation en tenseur (28x28)")
    st.image(tensor_image.squeeze().cpu().numpy(), caption="Tensor final (28x28)", clamp=True)
    return tensor_image

def classify_image(tensor_image):
    """
    Classification de l'image à l'aide du modèle CNN.
    """
    with torch.no_grad():
        output = cnn_model(tensor_image)
        probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
        predicted = torch.argmax(output, dim=1).item()
    return predicted, probabilities

def reconstruct_image(tensor_image):
    """
    Reconstruction de l'image à l'aide de l'AutoEncoder.
    """
    with torch.no_grad():
        # Passer l'image dans l'AutoEncoder
        reconstructed = autoencoder(tensor_image).squeeze(0).cpu().numpy()

        # S'assurer que l'image est dans la plage [0, 1]
        reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())

        # Ajouter des canaux RGB en répétant les valeurs
        reconstructed_rgb = np.repeat(reconstructed, 3, axis=0)  # Dupliquer les canaux
        reconstructed_rgb = np.transpose(reconstructed_rgb, (1, 2, 0))  # Reformatage en (H, W, C)
    return reconstructed_rgb


# Interface utilisateur avec Streamlit
st.title("Reconnaissance et Reconstruction de Chiffres Manuscrits")
st.write("### Dessinez un chiffre dans la zone ci-dessous, puis cliquez sur **Prédire** pour voir la classification et la reconstruction.")

# Zone de dessin
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    # Convertir le dessin en image PIL
    image = Image.fromarray((canvas_result.image_data[:, :, 0] * 255).astype(np.uint8))
    
    # Bouton pour prédire
    if st.button("Prédire"):
        tensor_image = preprocess_image(image)

        # Classification
        predicted, probabilities = classify_image(tensor_image)
        st.write(f"### Classification : Le modèle prédit : **{predicted}**")
        st.write("#### Probabilités pour chaque classe :")
        st.dataframe(probabilities)

        # Reconstruction
        reconstructed_image = reconstruct_image(tensor_image)
        st.write("### Reconstruction de l'image avec l'AutoEncoder :")
        st.image(reconstructed_image, caption="Image reconstruite", clamp=True, use_column_width=True)


