# Generated by Django 2.1.7 on 2019-03-29 03:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0007_auto_20190328_0850'),
    ]

    operations = [
        migrations.AlterField(
            model_name='fileupload',
            name='result',
            field=models.TextField(default='', max_length=255),
        ),
    ]
