# Model/serializers.py
from rest_framework import serializers

class ImageSerializer(serializers.Serializer):
    image = serializers.ListField(
        child=serializers.FloatField(),  # Suponiendo que cada valor en la imagen sea un número flotante (puedes ajustarlo si es necesario)
        write_only=True  # Asegúrate de que esto no se devuelva en la respuesta
    )
