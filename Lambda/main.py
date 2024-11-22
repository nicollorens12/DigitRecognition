import base64
import json
import numpy as np
from PIL import Image
from io import BytesIO
from CustomNeuralNetwork import CustomNeuralNetwork

# Carga tu modelo
model = CustomNeuralNetwork("model.h5")

def preprocess_image(rgba_data):
    # (Tu lógica de preprocesamiento permanece igual)
    grayscale_image = np.dot(rgba_data[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    inverted_image = 255 - grayscale_image

    threshold = 50
    binary_image = (inverted_image > threshold).astype(np.uint8)

    coords = np.column_stack(np.where(binary_image > 0))
    if coords.size == 0:
        raise ValueError("No se detectó ningún dígito en la imagen")

    top_left = coords.min(axis=0)
    bottom_right = coords.max(axis=0)

    cropped_image = inverted_image[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

    pil_cropped = Image.fromarray(cropped_image, 'L')
    width, height = pil_cropped.size
    aspect_ratio = width / height

    if width > height:
        new_width = 20
        new_height = int(round(new_width / aspect_ratio))
    else:
        new_height = 20
        new_width = int(round(new_height * aspect_ratio))

    resized_cropped = pil_cropped.resize((new_width, new_height), Image.LANCZOS)

    centered_image = Image.new('L', (28, 28), color=0)
    offset_x = (28 - new_width) // 2
    offset_y = (28 - new_height) // 2
    centered_image.paste(resized_cropped, (offset_x, offset_y))

    processed_image = np.array(centered_image) / 255.0
    return processed_image

def predict(event, context):
    try:
        # Extraer el cuerpo de la solicitud (imagen codificada en base64)
        body = event.get("body", "")
        if not body:
            return {
                "statusCode": 400,
                "body": json.dumps({"Error": "No image data found in the request."})
            }

        # Decodificar la imagen
        image_data = base64.b64decode(body)
        image = Image.open(BytesIO(image_data))

        # Validar tamaño de la imagen (280x280 RGBA)
        if image.size != (280, 280) or image.mode != "RGBA":
            return {
                "statusCode": 400,
                "body": json.dumps({"Error": "Invalid image shape, must be 280x280 with RGBA."})
            }

        rgba_data = np.array(image, dtype=np.uint8)

        # Preprocesar la imagen
        rgba_data[rgba_data[..., 3] == 0] = [255, 255, 255, 255]
        processed_image = preprocess_image(rgba_data)

        # Aplanar la imagen para el modelo
        flattened_image = processed_image.flatten()

        # Hacer la predicción
        prediction = model.predict(flattened_image)

        # Responder con el resultado
        return {
            "statusCode": 200,
            "body": json.dumps({"Prediction": str(prediction)})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"Error": str(e)})
        }
