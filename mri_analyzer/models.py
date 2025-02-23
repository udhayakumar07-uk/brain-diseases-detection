# mri_analyzer/models.py

from django.db import models
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

class MRIImage(models.Model):
    image = models.ImageField(upload_to='mri_images/')
    prediction = models.CharField(max_length=255, null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    disease_type = models.CharField(max_length=255, null=True, blank=True)
    affected_region = models.JSONField(null=True, blank=True)  # Store bounding box coordinates
    
    def __str__(self):
        return f"MRI Image {self.id}"

class DiseasePredictor:
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.models = {
            'alzheimers': load_model('models/alzheimers_model.h5'),
            'stroke': load_model('models/stroke_model.h5'),
            'tumor': load_model('models/tumor_model.h5')
        }
        self.class_names = {
            'alzheimers': ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented'],
            'stroke': ['Haemorrhagic', 'Ischemic', 'Normal'],
            'tumor': ['glioma', 'meningioma', 'notumor', 'pituitary']
        }
        # Force build each model
        dummy_input = np.zeros((1, self.img_size, self.img_size, 3), dtype=np.float32)
        for model in self.models.values():
            model.predict(dummy_input)

    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(self.img_size, self.img_size))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def get_bounding_box(self, heatmap, threshold=0.5):
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Threshold the heatmap
        binary = (heatmap > threshold).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        return (x, y, w, h)

    def get_gradcam(self, img_array, model):
        if not model.built:
            model.build((None, self.img_size, self.img_size, 3))
        _ = model(img_array)

        last_conv_layer = None
        for layer in model.layers[::-1]:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        if last_conv_layer is None:
            raise ValueError("No convolutional layer found in the model.")

        grad_model = tf.keras.models.Model(
            model.inputs,
            [last_conv_layer.output, model.layers[-1].output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            tape.watch(conv_outputs)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None:
            grads = tf.zeros_like(conv_outputs)
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        return heatmap.numpy()

    def predict_and_visualize(self, image_path):
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        processed_img = self.preprocess_image(image_path)
        
        results = {}
        best_prediction = {'confidence': 0}
        
        for disease_type, model in self.models.items():
            prediction = model.predict(processed_img)
            pred_class = np.argmax(prediction[0])
            confidence = prediction[0][pred_class]
            class_name = self.class_names[disease_type][pred_class]
            
            normal_classes = ['Normal', 'notumor', 'NonDemented']
            if confidence > best_prediction['confidence'] and class_name not in normal_classes:
                best_prediction = {
                    'disease_type': disease_type,
                    'class': class_name,
                    'confidence': confidence
                }
            
            if confidence > 0.5 and class_name not in normal_classes:
                # Generate heatmap
                heatmap = self.get_gradcam(processed_img, model)
                
                # Get bounding box
                bbox = self.get_bounding_box(heatmap)
                
                # Resize heatmap to original image size
                heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
                heatmap_display = np.uint8(255 * heatmap_resized)
                heatmap_colored = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET)
                
                # Create visualization with both heatmap and bounding box
                superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
                
                # Draw bounding box if found
                if bbox is not None:
                    x, y, w, h = bbox
                    # Scale bbox to original image size
                    scale_x = original_img.shape[1] / self.img_size
                    scale_y = original_img.shape[0] / self.img_size
                    x, w = int(x * scale_x), int(w * scale_x)
                    y, h = int(y * scale_y), int(h * scale_y)
                    cv2.rectangle(superimposed_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Store bbox coordinates
                    best_prediction['bbox'] = {
                        'x': x, 'y': y, 'width': w, 'height': h
                    }
            else:
                superimposed_img = original_img
            
            results[disease_type] = {
                'class': class_name,
                'confidence': confidence,
                'visualization': superimposed_img
            }
        
        return results, best_prediction