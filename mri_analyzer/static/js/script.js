// static/js/script.js

document.addEventListener('DOMContentLoaded', function() {
    const mriForm = document.getElementById('mri-form');
    const mriInput = document.getElementById('mri-image');
    const scanBtn = document.getElementById('scan-btn');
    const resultsContainer = document.querySelector('.results-container');
    const scanningIndicator = document.querySelector('.scanning-indicator');
    const originalMRI = document.getElementById('original-mri');
    const analyzedMRI = document.getElementById('analyzed-mri');
    const predictionText = document.getElementById('prediction-text');
    const diseaseTypeText = document.getElementById('disease-type-text');
    const confidenceText = document.getElementById('confidence-text');

    mriInput.addEventListener('change', function() {
        scanBtn.disabled = !this.files.length;
        if (this.files.length) {
            const file = this.files[0];
            originalMRI.src = URL.createObjectURL(file);
            document.querySelector('.file-label span').textContent = file.name;
        }
    });

    mriForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        scanBtn.disabled = true;
        scanBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        
        resultsContainer.style.display = 'block';
        scanningIndicator.style.display = 'block';
        
        try {
            const response = await fetch('/upload/', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Update results
                predictionText.textContent = data.prediction;
                diseaseTypeText.textContent = data.disease_type;
                confidenceText.textContent = data.confidence;
                
                // Update images
                originalMRI.src = data.original_image;
                analyzedMRI.src = data.analyzed_image;
                
                // Hide scanning indicator and show results
                scanningIndicator.style.display = 'none';
                resultsContainer.scrollIntoView({ behavior: 'smooth' });
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing the image.');
        } finally {
            scanBtn.disabled = false;
            scanBtn.innerHTML = '<i class="fas fa-brain"></i> Scan MRI';
        }
    });
});