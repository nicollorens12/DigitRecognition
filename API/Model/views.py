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

model = CustomNeuralNetwork("Model/model.h5")

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
        400: 'Invalid image shape, must be 28x28.',
        405: 'Invalid HTTP method. Use POST.',
        500: 'Internal server error'
    }
)
@api_view(['POST'])
def predict_digit(request):
    if request.method == 'POST':
        try:
            # Validar los datos
            serializer = ImageSerializer(data=request.data)
            if serializer.is_valid():
                rgba_data = np.array(serializer.validated_data['image'], dtype=np.uint8)

                if len(rgba_data) != 280 * 280 * 4: 
                    return JsonResponse({'error': 'Invalid image shape, must be 280x280 with RGBA'}, status=400)

                rgba_data = rgba_data.reshape((280, 280, 4))

                rgba_data[rgba_data[..., 3] == 0] = [255, 255, 255, 255]

                #DEBUG = os.getenv('DEBUG')
                #if DEBUG:
                #    raw_image = Image.fromarray(rgba_data, 'RGBA')
                #    raw_image.save('Model/data/raw_image_with_white_background.png')

                grayscale_image = np.dot(rgba_data[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
                inverted_grayscale_image = 255 - grayscale_image
                # Reescalar a 28x28
                pil_image = Image.fromarray(inverted_grayscale_image, 'L')
                resized_image = pil_image.resize((28, 28), Image.LANCZOS)

                # Guardar la imagen procesada
                #if DEBUG:
                #    resized_image.save('Model/data/resized_image.png')

                flattened_image = np.array(resized_image).flatten()
                normalized_image = flattened_image / 255.0

                prediction = model.predict(normalized_image)

                return JsonResponse({'prediction': int(prediction)})
            else:
                return JsonResponse({'error': 'Invalid data, check the image format'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid HTTP method. Use POST.'}, status=405)


