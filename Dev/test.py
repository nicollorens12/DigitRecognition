from tensorflow.keras.datasets import mnist

# Cargar MNIST y obtener una muestra
(_, _), (x_test, _) = mnist.load_data()
sample_image = x_test[0]  # Tomar la primera imagen

# Preprocesar la imagen como se haría en el frontend
image_data = sample_image.flatten().tolist()

# Enviar a la API
import requests
response = requests.post(
    "http://localhost:8000/model/predict/",
    json={"image": image_data}
)
print(response.json())