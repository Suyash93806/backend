from django.db import models
from django.contrib.auth.models import User


class UserItem(models.Model):
    auth = models.OneToOneField(User, on_delete=models.CASCADE, null=True)
    name = models.CharField(max_length=100,null=False,blank=True)
    email = models.EmailField(max_length=100,null=False,blank=True)

    def __str__(self):
        return f"{self.email}"

class Informn(models.Model):
    emotion=models.CharField(max_length=200,null=True)
    description=models.TextField(null=True)
    

    def __str__(self):
        return f"{self.description}"

