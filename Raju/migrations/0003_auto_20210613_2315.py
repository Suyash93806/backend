# Generated by Django 3.2.3 on 2021-06-13 17:45

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('Raju', '0002_auto_20210604_2214'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='secondmodel',
            name='datetime',
        ),
        migrations.RemoveField(
            model_name='secondmodel',
            name='image',
        ),
    ]