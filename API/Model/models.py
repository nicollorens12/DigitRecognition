from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .nn import CustomNeuralNetwork
import numpy as np
import json

# Carga los parámetros del modelo al iniciar la vista
model = CustomNeuralNetwork(layer_dims=[784, 128, 16, 16, 10])
model.load_params('neuralnet/model_params.json')  # Ruta relativa al archivo JSON

@csrf_exempt
def predict_digit(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)  # Suponemos que los datos vienen en formato JSON
            input_array = np.array(data['digit'], dtype=np.float32).reshape(784, 1)  # Asegúrate de la dimensión correcta
            
            prediction = model.predict(input_array)
            return JsonResponse({'prediction': int(prediction)})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)
