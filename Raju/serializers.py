from rest_framework import serializers
from . import models
from django.contrib.auth.models import User

class new(serializers.ModelSerializer):

    class Meta:
        model = models.UserItem
        fields = ('name','email')
    
    