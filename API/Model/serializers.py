from rest_framework import serializers

class ImageSerializer(serializers.Serializer):
    image = serializers.ListField(
        child=serializers.IntegerField(),
        allow_empty=False
    )
