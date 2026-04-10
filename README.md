
## Auteure

| Champ | Information |
|-------|-------------|
| **Nom** | Awa FAYE |
| **Etablissement** | Universite Iba Der THIAM de Thies |
| **Formation** | UFR SES/SET — Master SDA (Option IA) |
| **Annee academique** | 2025-2026 |
| **Email** | awa.faye16@univ-thies.sn |
| **GitHub** | [awafaye16-ctl](https://github.com/awafaye16-ctl) |

---


---

## 1. Presentation du projet

**Lafayette Medical AI** est une application web complete developpee avec
**Flask (Python)** pour la classification histopathologique du cancer du sein.
Elle integre un modele deep learning **ResNet-50**, une visualisation
**Grad-CAM** et une generation de rapports medicaux automatiques via
**Groq API (Llama 3.1 70B)** — entierement gratuit.

### Pourquoi Flask ?

Flask a ete choisi pour 3 raisons fondamentales :

**1. Integration native avec PyTorch**
Flask et PyTorch sont tous les deux en Python. Le modele `model.pth` est
charge directement dans `app.py` sans aucune couche intermediaire. Pas
besoin de microservice separe, pas de communication inter-processus.

**2. Simplicite de developpement**
En Flask, une route s'ecrit en 5 lignes. L'ensemble de la logique
(chargement modele, prediction, Grad-CAM, LLM, sauvegarde) est dans
un seul fichier `app.py` de ~350 lignes, facilement maintenable.

**3. Deploiement simple**
Une seule commande `python app.py` lance l'application complete.
Le deploiement Docker se fait avec un seul `Dockerfile` et une seule
image contenant Python + PyTorch + Flask.

### Contexte medical

Le cancer du sein est la premiere cause de mortalite feminine par cancer
dans le monde. L'analyse histopathologique manuelle est longue, subjective
et dependante de la disponibilite d'un pathologiste expert. Ce systeme :

- **Classifie** automatiquement les images de tissu mammaire H&E
- **Interprete** ses decisions via Grad-CAM (zones actives visualisees)
- **Communique** les resultats en langage naturel (rapports LLM)
- **Reproduit** ses performances (validation K-Fold rigoureuse)

### Dataset BreaKHis 400x

| Propriete | Valeur |
|-----------|--------|
| Magnification | 400x |
| Images benignes | 547 (32%) |
| Images malignes | 1 146 (68%) |
| Total | 1 693 images |
| Coloration | Hematoxyline & Eosine (H&E) |
| Train+Val | 1 354 images (K-Fold) |
| Test set isole | 339 images (jamais vues) |

---

## 2. Fonctionnalites

### F1 — Classification IA avec TTA (Test-Time Augmentation)

**Qu'est-ce que c'est ?**
La prediction utilise le modele ResNet-50 entraine sur BreaKHis 400x.
Pour ameliorer la stabilite, on utilise le TTA : au lieu d'une seule
prediction, on genere 10 versions legerement differentes de l'image
(recadrages et retournements aleatoires) et on moyenne les probabilites.

**Pourquoi TTA ?**
Sans TTA, une image cadrée légèrement différemment peut donner
51% malin vs 49% benin — une prediction instable. Avec TTA (10
augmentations), la variance est reduite d'un facteur sqrt(10) = 3.16x.

**Seuil optimise 0.367 (pas 0.5)**
Le seuil par defaut de 0.5 detecte seulement 65.1% des cancers malins.
En abaissant le seuil a 0.367 (determine par analyse ROC), on detecte
90.4% des malins au prix de quelques biopsies supplementaires.

**Resultat affiche :**
- Badge BENIN (vert) ou MALIN (rouge) avec animation
- Pourcentage de confiance avec jauge animee
- Temps d'inference en millisecondes
- Seuil utilise (0.367)

### F2 — Visualisation Grad-CAM

**Qu'est-ce que Grad-CAM ?**
Grad-CAM (Gradient-weighted Class Activation Mapping) est une technique
d'interpretabilite qui produit une carte thermique (heatmap) montrant
quelles zones de l'image ont le plus influence la decision du modele.

**Comment ca marche dans app.py ?**
1. Un hook forward sur `layer4[-1].conv3` capture les activations (7x7x2048)
2. Un hook backward capture les gradients lors de la retropropagation
3. Les poids = moyenne spatiale des gradients par canal
4. CAM = ReLU(somme ponderee des activations par les poids)
5. Interpolation bilineaire 7x7 → 224x224
6. Colorisation COLORMAP_JET + superposition 60% image / 40% heatmap

**Interpretation des couleurs :**
```
Bleu fonce   → Zone peu importante pour la decision
Cyan/Vert    → Zone moyennement active
Jaune/Orange → Zone importante
Rouge vif    → Zone tres critique (determinante)
```

**Pourquoi c'est important ?**
Sans Grad-CAM, le modele est une boite noire. Un medecin ne peut pas
valider une decision qu'il ne comprend pas. Sur 4/5 images testees,
les zones rouges correspondent aux noyaux cellulaires atypiques —
les structures biologiquement pertinentes pour le diagnostic.

### F3 — Rapport medical par LLM (Groq + Llama 3.1 70B)

**Qu'est-ce que Groq ?**
Groq est un service d'inference ultra-rapide pour les LLM (Large Language
Models). Il donne acces au modele Llama 3.1 70B de Meta entierement
gratuitement (15 requetes/minute, sans carte bancaire).

**Pourquoi Groq plutot que Gemini ou OpenAI ?**
- Gratuit sans limite stricte pour la recherche
- Reponse en moins d'une seconde
- Cle API courte (format gsk_xxx) — pas de probleme Windows CMD
- Qualite medicale excellente avec Llama 3.1 70B

**Structure du rapport genere :**
1. Resultat de l'analyse IA et niveau de confiance
2. Signification clinique du diagnostic
3. Recommandation au medecin referent
4. Avertissement medical obligatoire

### F4 — Dashboard de performances

Page dediee affichant :
- 4 cartes metriques : AUC-ROC, Accuracy, Recall Malin, F1
- Courbe ROC avec point seuil 0.5 et point optimal 0.367 annotés
- Graphique K-Fold (5 folds) avec barres Accuracy + AUC
- Matrice de confusion coloree (VP/VN/FP/FN)
- Intervalles de confiance 95%

### F5 — Historique et export CSV

- Sauvegarde automatique dans `predictions.json` apres chaque analyse
- Tableau pagine avec filtres Benin/Malin
- Miniatures cliquables (image originale + heatmap)
- Export CSV telechargeable

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
git clone https://github.com/awafaye16-ctl/breakhis-flask.git
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
#   B r e a K H i s - C N N _ w i t h _ F l a s k _ V 2  
 