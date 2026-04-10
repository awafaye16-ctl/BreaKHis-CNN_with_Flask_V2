# Lafayette Medical AI

Application de classification histopathologique du cancer du sein avec **PyTorch réel**, **Grad-CAM véritable** et le modèle **best_model_resnet50_v2**.

> **Précision. Interprétabilité. Soins.**

---

## Auteur

**Awa FAYE** - Master SDA (Option IA) - Université Iba Der THIAM de Thies - 2025-2026

---

## Test et Validation

### **Test 1 : Vérification PyTorch Réel**
```bash
# Vérifier que PyTorch utilise votre modèle
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import os; print('Modèle présent:', os.path.exists('best_model_resnet50_v2.pth'))"

# Résultat attendu :
# PyTorch: 2.1.0+cpu
# Modèle présent: True
```

### **Test 2 : Vérification Grad-CAM Réel**
```bash
# Vérifier Grad-CAM avec hooks PyTorch
python -c "from app_final import load_model; model, grad_cam = load_model(); print('Grad-CAM:', 'Prêt' if grad_cam else 'Non prêt')"

# Résultat attendu :
# Grad-CAM: Prêt
```

### **Test 3 : Vérification Groq LLM Réel**
```bash
# Vérifier la clé API Groq
python -c "from dotenv import load_dotenv; import os; load_dotenv(); key = os.environ.get('GROQ_API_KEY', ''); print('Groq:', 'Disponible' if key.startswith('gsk_') else 'Non disponible')"

# Résultat attendu :
# Groq: Disponible
```

### **Test 4 : Test Complet d'Analyse**
```bash
# Démarrer l'application
python app-final.py

# Accéder à http://localhost:5000
# Uploader une image histopathologique H&E
# Vérifier les résultats :
# - Prédiction PyTorch réelle
# - Grad-CAM véritable
# - Rapport médical Groq LLM
```

---

## Démarrage Rapide

### 1. Installation

```bash
# Cloner et installer
git clone https://github.com/awafaye16-ctl/BreaKHis-CNN_with_Flask_V2.git
cd lafayette-medical-ai
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Créer fichier .env
notepad .env
# Ajouter :
GROQ_API_KEY=gsk_votre_clé_ici
```

### 3. Lancement

```bash
# Lancer l'application
python app-final.py
```

**Accès :** http://localhost:5000

---

## Fonctionnalités

### **PyTorch Réel** 
- **Modèle :** `best_model_resnet50_v2.pth` (24.6M paramètres)
- **Architecture :** ResNet-50 personnalisé
- **Performance :** 86.1% accuracy, AUC 0.9655
- **Inférence :** ~200ms sur CPU

### **Grad-CAM Véritable**
- **Interprétabilité :** Zones d'intérêt basées sur gradients réels
- **Visualisation :** Heatmaps superposées avec OpenCV
- **Médical :** Compréhension transparente des décisions
- **Hooks :** Forward/Backward sur layer4[-1]

### **Groq LLM**
- **Modèle :** Mixtral-8x7b-32768
- **Rapports :** Générés automatiquement en français
- **Vision :** Analyse d'images base64
- **Fallback :** Templates si API indisponible

---

## Architecture

```
lafayette-medical-ai/
|
|-- app-final.py              # Application principale (PyTorch réel)
|-- best_model_resnet50_v2.pth # Votre modèle entraîné (98 MB)
|-- requirements.txt          # Dépendances
|-- templates/                # Pages HTML
|    |-- index.html           # Page d'analyse
|    |-- dashboard.html       # Tableau de bord
|    |-- history.html         # Historique
|-- static/                   # Uploads + heatmaps
|-- data/                     # Métriques
```

---

## Utilisation

### Page Analyse
1. **Upload** d'image histopathologique H&E
2. **Analyse** automatique (~200ms)
3. **Résultats** :
   - Prédiction PyTorch réelle
   - Grad-CAM véritable
   - Rapport médical Groq

### API Endpoints

```bash
# Prédiction
POST /api/predict
Content-Type: multipart/form-data
Body: file=<image.png>

# Health check
GET /health

# Métriques
GET /api/metrics
```

---

## Performance

| Métrique | Valeur | Test |
|----------|--------|------|
| **Accuracy** | 86.1% | Dataset BreaKHis |
| **AUC-ROC** | 0.9655 | Validation croisée |
| **Recall Malin** | 90.4% | Seuil optimisé |
| **Temps inférence** | ~200ms | CPU PyTorch |
| **Taille modèle** | 98 MB | best_model_resnet50_v2.pth |

---

## Validation Technique

### **Composants Réels Confirmés**
- **PyTorch** : `model(input_tensor)` avec vrais tenseurs
- **Grad-CAM** : Hooks PyTorch + calculs gradients
- **Groq** : API REST Mixtral-8x7b-32768
- **OpenCV** : Superposition heatmaps COLORMAP_JET

### **Conditions d'Utilisation**
```python
# PyTorch réel si modèle chargé
if model is not None:
    outputs = model(input_tensor)
    # Calculs PyTorch authentiques

# Grad-CAM réel si hooks prêts
if grad_cam is not None:
    cam, class_idx = grad_cam.generate(input_tensor)
    # Grad-CAM authentique

# Groq réel si clé API présente
if groq_client and GROQ_API_KEY:
    # API Groq authentique
```

---

## Dépannage

### Modèle non chargé
```bash
# Vérifier la présence du modèle
dir best_model_resnet50_v2.pth
# Résultat attendu : 98 MB
```

### Erreur NumPy
```bash
# Réinstaller compatibilité
pip install numpy==1.26.4
```

### Port 5000 utilisé
```bash
# Changer port dans app-final.py
app.run(port=5001)
```

### Groq non disponible
```bash
# Vérifier clé API
type .env
# Doit contenir : GROQ_API_KEY=gsk_xxx
```

---

## Versions Disponibles

| Fichier | PyTorch | Grad-CAM | Groq | Usage |
|---------|---------|----------|------|-------|
| **app-final.py** | **Réel** | **Vrai** | **Réel** | **Production** |
| app-groq.py | Non | Simulé | Réel | Développement |
| app-minimal.py | Non | Non | Non | Test |

---

## Avertissement Médical

> **USAGE RECHERCHE UNIQUEMENT** - Cette application ne remplace pas l'avis d'un pathologiste certifié. Toute décision clinique doit être validée par un professionnel de santé qualifié.

---

**Développé avec PyTorch, Grad-CAM, Groq et détermination.**

*awa.faye16@univ-thies.sn | github.com/awafaye16-ctl*
