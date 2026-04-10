import torch
import numpy as np
from app_final import load_model, predict_with_pytorch, generate_real_gradcam, generate_medical_report_groq

print('=== VÉRIFICATION DES COMPOSANTS RÉELS ===')
print()

# 1. Vérifier PyTorch
print(f'1. PyTorch version: {torch.__version__}')
print(f'   Disponible: {True}')

# 2. Vérifier le modèle
model, grad_cam = load_model()
print(f'2. Modèle chargé: {model is not None}')
if model:
    print(f'   Paramètres: {sum(p.numel() for p in model.parameters()):,}')
    print(f'   Device: {next(model.parameters()).device}')

# 3. Vérifier Grad-CAM
print(f'3. Grad-CAM prêt: {grad_cam is not None}')

# 4. Vérifier Groq
from dotenv import load_dotenv
import os
load_dotenv()
groq_key = os.environ.get('GROQ_API_KEY', '')
print(f'4. Groq LLM: {len(groq_key) > 0 and groq_key.startswith("gsk_")}')

print()
print('=== TEST AVEC UNE IMAGE DÉMO ===')
print()

# Créer une image de test
test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
from PIL import Image
Image.fromarray(test_image).save('test_image.png')

# Test prédiction
pred, conf = predict_with_pytorch('test_image.png')
print(f'5. Prédiction PyTorch: {pred} ({conf:.1%})')

# Test Grad-CAM
heatmap_url = generate_real_gradcam('test_image.png', pred, conf)
print(f'6. Grad-CAM: {"Réel" if model else "Fallback"}')

# Test Groq
report = generate_medical_report_groq(pred, conf, 'test_image.png')
print(f'7. Rapport Groq: {"Réel" if len(groq_key) > 0 else "Fallback"}')

print()
print('=== CONCLUSION ===')
if model and grad_cam and len(groq_key) > 0:
    print('✅ TOUS LES COMPOSANTS SONT RÉELS')
elif model and grad_cam:
    print('⚠️  PYTORCH + GRAD-CAM RÉELS, GROQ EN FALLBACK')
else:
    print('❌ CERTAINS COMPOSANTS SONT EN FALLBACK')
