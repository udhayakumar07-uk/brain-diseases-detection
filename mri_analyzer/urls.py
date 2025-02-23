# mri_analyzer/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('upload/', views.upload_mri, name='upload'),  # Changed from 'upload_mri' to 'upload'
]