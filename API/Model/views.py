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
import pickle

# Cargar el modelo preentrenado
pretrained_params = CustomNeuralNetwork.load_params('Model/model.h5')
model = CustomNeuralNetwork(pretrained_params=pretrained_params)

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
            # Usamos el serializer para validar los datos
            serializer = ImageSerializer(data=request.data)
            if serializer.is_valid():
                image = np.array(serializer.validated_data['image'], dtype=np.float32)
                
                # Verificar si la imagen está aplanada y convertirla a la forma 28x28
                if image.shape != (784,):  # Si la imagen está aplanada
                    return JsonResponse({'error': 'Invalid image shape, must be 28x28.'}, status=400)
                    
                print('Image shape:', image.shape)
                # Normalizar los datos si es necesario
                image = image / 255.0  # Asegurarse de que los valores estén entre 0 y 1
                
                params = None

                with open("Model/params.pkl", "rb") as file:
                    params = pickle.load(file)

                # Realizar la predicción
                prediction = model.predict(image,params)  # Aplanar si necesario
                
                return JsonResponse({'prediction': int(prediction)})
            else:
                return JsonResponse({'error': 'Invalid data, check the image format'}, status=400)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Invalid HTTP method. Use POST.'}, status=405)
