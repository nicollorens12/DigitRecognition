# Model/views.py
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .CustomNeuralNetwork import CustomNeuralNetwork
import numpy as np
import json
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from .serializers import ImageSerializer  # Importa el serializer
from PIL import Image
import os
from scipy.ndimage import measurements

model = CustomNeuralNetwork("Model/model.h5")

def preprocess_image(rgba_data):
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




@swagger_auto_schema(
    method='post',
    request_body=ImageSerializer,  # Usar el serializer aquí
    responses={
        200: openapi.Response('Prediction result', openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'prediction': openapi.Schema(type=openapi.TYPE_INTEGER)
            }
        )),
        400: 'Invalid image shape, must be 280x280 with RGBA.',
        405: 'Invalid HTTP method. Use POST.',
        500: 'Internal server error'
    }
)
@api_view(['POST'])
def predict_digit(request):
    if request.method == 'POST':
        try:
            serializer = ImageSerializer(data=request.data)
            if serializer.is_valid():
                rgba_data = np.array(serializer.validated_data['image'], dtype=np.uint8)

                if len(rgba_data) != 280 * 280 * 4: 
                    return JsonResponse({'error': 'Invalid image shape, must be 280x280 with RGBA'}, status=400)

                rgba_data = rgba_data.reshape((280, 280, 4))

                rgba_data[rgba_data[..., 3] == 0] = [255, 255, 255, 255]

                #raw_image = Image.fromarray(rgba_data, 'RGBA')
                #raw_image.save('Model/data/raw_image.png')

                processed_image = preprocess_image(rgba_data)

                #preprocessed_image_pil = Image.fromarray((processed_image * 255).astype(np.uint8), 'L')
                #preprocessed_image_pil.save('Model/data/preprocessed_image.png')
                
                flattened_image = processed_image.flatten() 

                prediction = model.predict(flattened_image)

                return JsonResponse({'prediction': int(prediction)})
            else:
                return JsonResponse({'error': 'Invalid data, check the image format'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid HTTP method. Use POST.'}, status=405)



