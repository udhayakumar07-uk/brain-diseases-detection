# Generated by Django 5.1.6 on 2025-02-23 02:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mri_analyzer', '0002_remove_mriimage_uploaded_at_mriimage_disease_type_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='mriimage',
            name='affected_region',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
