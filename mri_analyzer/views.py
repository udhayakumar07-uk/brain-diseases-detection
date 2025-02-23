# mri_analyzer/views.py

from django.shortcuts import render
from django.http import JsonResponse
from .models import MRIImage, DiseasePredictor
import cv2
import numpy as np
import base64
import os
from django.conf import settings

predictor = DiseasePredictor()

def home(request):
    return render(request, 'mri_analyzer/home.html')

def about(request):
    return render(request, 'mri_analyzer/about.html')

def contact(request):
    return render(request, 'mri_analyzer/contact.html')

def upload_mri(request):
    if request.method == 'POST' and request.FILES.get('mri_image'):
        mri_image = MRIImage(image=request.FILES['mri_image'])
        mri_image.save()
        
        # Get predictions and visualizations
        results, best_prediction = predictor.predict_and_visualize(mri_image.image.path)
        
        # Save the visualization
        for disease_type, result in results.items():
            if disease_type == best_prediction['disease_type']:
                vis_path = os.path.join(settings.MEDIA_ROOT, f'visualizations/{mri_image.id}.jpg')
                os.makedirs(os.path.dirname(vis_path), exist_ok=True)
                cv2.imwrite(vis_path, cv2.cvtColor(result['visualization'], cv2.COLOR_RGB2BGR))
                vis_url = f'/media/visualizations/{mri_image.id}.jpg'
                break
        
        # Update the MRIImage instance
        mri_image.prediction = best_prediction['class']
        mri_image.confidence = best_prediction['confidence'] * 100
        mri_image.disease_type = best_prediction['disease_type']
        mri_image.save()
        
        return JsonResponse({
            'status': 'success',
            'prediction': best_prediction['class'],
            'disease_type': best_prediction['disease_type'],
            'confidence': f"{best_prediction['confidence'] * 100:.2f}%",
            'original_image': mri_image.image.url,
            'analyzed_image': vis_url
        })
    
    return render(request, 'mri_analyzer/upload.html')
