---
title: Détection d'avions militaires
description: Détection d'avions militaires sur des images et vidéos à l'aide de YOLOv8.
author: <author_id>
date: 2024-06-01 11:33:00 +0800
categories: [Projet Personnel, Deep Learning]
tags: [projet, computer vision, YOLOv8, Deep Learning, Object Detection, detection, classification]
pin: true
math: true
mermaid: true
image:
  path: /assets/img/AirCraft/presentation.jpg
  alt: F18 airplanes detection
---

# Détection d'avions militaires sur des images et vidéos à l'aide de YOLOv8

## Introduction

Dans ce projet, j'ai utilisé un modèle YOLOv8 pour détecter et classifier les avions militaires sur des images et des vidéos. L'objectif est de montrer la puissance de l'Object Detection dans le domaine de la Computer Vision, en particulier pour la reconnaissance d'objets complexes et en mouvement, et nottament pour des objets ayant des caractéristiques similaires comme les avions militaires.

## YOLOv8

YOLOv8 est un modèle d'Object Detection basé sur le réseau de neurones YOLO (You Only Look Once). Il s'agit de la huitième version de ce modèle, qui a été amélioré pour être plus rapide et plus précis que ses prédécesseurs. Développé par Ultralytics, YOLOv8 offre des améliorations significatives en termes de précision et de vitesse par rapport à ses prédécesseurs.

### Caractéristiques Principales de YOLOv8
- **Détection en temps réel** : Capable de traiter des vidéos en direct pour détecter des objets instantanément.
- **Haute Précision** : Utilise des techniques avancées de deep learning pour fournir des prédictions précises.
- **Efficacité** : Conçu pour être utilisé même sur des machines avec des ressources limitées, comme des ordinateurs portables sans GPU puissant.

### Avantages de YOLOv8
- **Vitesse** : Optimisé pour la rapidité, ce qui est essentiel pour des applications telles que la surveillance aérienne.
- **Polyvalence** : Peut être appliqué à divers types d'objets et contextes, y compris la détection d'avions dans des images et des vidéos.
- **Facilité d'Utilisation** : Intégré avec des outils de développement populaires et bien documenté, ce qui facilite sa mise en œuvre.

En utilisant YOLOv8, ce projet vise à démontrer comment les technologies modernes de vision par ordinateur peuvent être appliquées efficacement pour la détection d'objets spécifiques dans des images et des vidéos, contribuant ainsi à des domaines tels que la sécurité et la surveillance aérienne.

## Dataset

Nous disposons d'un riche ensemble de données comprenant 14 500 images, chacune contenant un ou plusieurs avions. Pour chaque image, un fichier CSV associé fournit des annotations détaillées des avions présents, incluant les coordonnées de leurs positions (xmin, ymin, xmax, ymax) ainsi que leur classification.

### Exemple d'Images du Dataset

Voici quelques exemples d'images du dataset utilisé pour entraîner et tester le modèle YOLOv8, avec leurs annotations correspondantes :

![C5](/assets/img/AirCraft/C5.jpg)

Cette image montre un avion C5 Galaxy, un avion de transport militaire lourd utilisé par l'US Air Force. Les avions militaires peuvent avoir des formes et des tailles variées, ce qui rend leur détection et classification difficiles pour les modèles d'Object Detection.  

![Mirage2000](/assets/img/AirCraft/Mirage2000.jpg)

Cette image montre un avion Mirage 2000 qui est un avion de chasse conçu par la société française Dassault Aviation, à la fin des années 1970. Le Mirage 2000 est principalement utilisé par l'Armée de l'air française qui en a reçu 315 exemplaires, tandis que 286 autres ont été exportés vers huit pays différents.  

Dans le set de données, il y a tout types d'image, des images plus ou moins claires, des images avec un ou plusieurs avions, des images avec des avions de différentes classes, etc. Celà permet de tester la capacité du modèle à détecter et classifier les avions dans des contextes variés.

### Exemple de Fichier CSV
| filename                           | width | height | class | xmin | ymin | xmax | ymax |
|------------------------------------|-------|--------|-------|------|------|------|------|
| 000aa01b25574f28b654718db0700f72   | 2048  | 1365   | F35   | 852  | 177  | 1998 | 503  |
| 000aa01b25574f28b654718db0700f72   | 2048  | 1365   | JAS39 | 169  | 769  | 549  | 893  |
| 000aa01b25574f28b654718db0700f72   | 2048  | 1365   | JAS39 | 125  | 908  | 440  | 1009 |
| 000aa01b25574f28b654718db0700f72   | 2048  | 1365   | B52   | 277  | 901  | 1288 | 1177 |

Ces données fournissent une base solide pour entraîner et tester notre modèle YOLOv8, en permettant de reconnaître avec précision divers types d'avions militaires dans des contextes variés.

## Entrainement du modèle

Pour entrainer un modèle YOLOv8, il est essentiel de disposer d'un ensemble de données étiquetées, qui servira de base pour l'apprentissage du modèle. Dans ce projet, nous avons utilisé un ensemble de données comprenant 14 500 images d'avions militaires, chacune étant annotée avec les coordonnées des avions présents et leur classification.

### Train Validation Test Split

Avant de commencer l'entraînement, nous avons divisé notre ensemble de données en trois parties distinctes : un ensemble d'entraînement (70%), un ensemble de validation (15%) et un ensemble de test (15%). Cette division nous permet de vérifier la performance du modèle sur des données inédites et de s'assurer qu'il généralise bien aux images qu'il n'a pas encore vues.

### Etapes de l'Entrainement

#### 1. **Configuration du Modèle**  
Un fichier de configuration YAML est créé pour spécifier :

Les chemins vers les ensembles d'entraînement, de validation et de test.
Le nombre de classes à détecter.
Les noms des classes.

Exemple de configuration :  

```yaml 
train: ../data/train.txt
val: ../data/val.txt
test: ../data/test.txt
nc: 3
names: ['F35', 'JAS39', 'B52']
```

#### 2. **Architecture du Modèle**  
YOLOv8 utilise une architecture de réseau de neurones convolutifs (CNN) avec plusieurs couches :  

*Convolutionnelles* : Pour extraire les caractéristiques des images.  
*Couches d'activation* : Pour introduire la non-linéarité.  
*Couches de mise en commun (pooling)* : Pour réduire la dimensionnalité.  
*Couches de prédiction* : Pour générer les prédictions des boîtes englobantes et des classes.  

Chaque couche est conçue pour capturer des informations spécifiques des images et les combiner pour produire des prédictions précises.  

#### #3. **Processus d'Entraînement**   
Pendant l'entraînement, le modèle passe par les étapes suivantes :   

*Propagation avant* : L'image passe à travers les couches du modèle, produisant des prédictions.  
*Calcul de la perte* : La différence entre les prédictions et les annotations réelles est calculée. La fonction de perte de YOLO combine les erreurs de classification, de localisation des boîtes et des objets manquants.  
*Propagation arrière* : Les gradients de la perte sont calculés et utilisés pour mettre à jour les poids du modèle via l'algorithme de descente de gradient.  
*Validation* : Après chaque epoch, le modèle est évalué sur l'ensemble de validation pour ajuster les hyperparamètres et éviter le surapprentissage.  


## Validation du Modèle

Après l'entraînement, le modèle est évalué sur l'ensemble de test pour mesurer sa performance. Les métriques suivantes sont utilisées pour évaluer la qualité des prédictions :

- **Box(P)**: Précision des détections des boîtes englobantes.
- **R**: Rappel, mesure de la capacité du modèle à retrouver toutes les instances pertinentes.
- **mAP50**: Mean Average Precision à 50% IoU (Intersection over Union), mesure la précision moyenne à un seuil de 50% d'IoU.
- **mAP50-95**: Mean Average Precision à différents seuils d'IoU, de 50% à 95%.

Ces métriques permettent d'évaluer la capacité du modèle à détecter et classifier les avions militaires avec précision et rappel, tout en minimisant les fausses détections et les faux négatifs.

Voici un exemple de code pour évaluer le modèle sur l'ensemble de test :

```python
!yolo val \
model='/kaggle/input/yolov7-military-plane/yolov8-m-best.pt' \
data='/kaggle/working/ultralytics/data/mad.yaml'\
augment \
batch=12 \
imgsz=1280
```
Ce code est utilisé pour évaluer le modèle sur l'ensemble de validation, en utilisant le modèle entraîné et les paramètres spécifiés dans le fichier de configuration YAML.

Voici les résultats de l'évaluation du modèle sur l'ensemble de validation :

| Class        | Images | Instances | Box(P) | R  | mAP50 | m |
|--------------|--------|-----------|--------|----|-------|---|
| all          | 2024   | 3408      | 0.944  | 0.86  | 0.938  | 0.89 |
| A10          | 48     | 82        | 0.961  | 0.927 | 0.953  | 0.902 |
| A400M        | 46     | 67        | 0.938  | 0.866 | 0.951  | 0.883 |
| AG600        | 34     | 35        | 0.994  | 1    | 0.995  | 0.981 |
| AV8B         | 37     | 65        | 0.963  | 0.985 | 0.986  | 0.97  |
| B1           | 49     | 67        | 0.929  | 0.91  | 0.951  | 0.912 |
| B2           | 52     | 68        | 0.95   | 0.844 | 0.96   | 0.855 |
| B52          | 50     | 64        | 0.979  | 0.922 | 0.967  | 0.929 |
| Be200        | 32     | 35        | 0.981  | 1    | 0.995  | 0.928 |
| C130         | 93     | 180       | 0.87   | 0.928 | 0.945  | 0.885 |
| C2           | 107    | 146       | 0.966  | 0.966 | 0.991  | 0.973 |
| C17          | 59     | 88        | 0.9    | 0.82  | 0.927  | 0.844 |
| C5           | 50     | 50        | 0.944  | 0.88  | 0.947  | 0.925 |
| E2           | 47     | 67        | 0.938  | 0.905 | 0.936  | 0.898 |
| E7           | 16     | 18        | 1      | 0.976 | 0.995  | 0.974 |
| EF2000       | 56     | 84        | 0.929  | 0.726 | 0.885  | 0.832 |
| F117         | 35     | 46        | 1      | 0.822 | 0.904  | 0.852 |
| F14          | 38     | 68        | 0.922  | 0.824 | 0.884  | 0.843 |
| F15          | 102    | 196       | 0.9    | 0.934 | 0.957  | 0.914 |
| F16          | 135    | 223       | 0.889  | 0.78  | 0.873  | 0.793 |
| F18          | 95     | 207       | 0.945  | 0.865 | 0.948  | 0.869 |
| F22          | 56     | 90        | 0.905  | 0.842 | 0.918  | 0.893 |
| F35          | 113    | 147       | 0.913  | 0.862 | 0.937  | 0.871 |
| F4           | 58     | 86        | 0.944  | 0.849 | 0.916  | 0.854 |
| JAS39        | 51     | 81        | 0.935  | 0.852 | 0.943  | 0.885 |
| MQ9          | 34     | 36        | 0.898  | 0.732 | 0.899  | 0.852 |
| Mig31        | 39     | 65        | 0.947  | 0.825 | 0.938  | 0.897 |
| Mirage2000   | 30     | 75        | 0.95   | 0.853 | 0.87   | 0.841 |
| P3           | 34     | 63        | 0.971  | 0.529 | 0.799  | 0.756 |
| RQ4          | 52     | 65        | 0.918  | 0.864 | 0.948  | 0.869 |
| Rafale       | 59     | 98        | 0.887  | 0.857 | 0.931  | 0.89  |
| SR71         | 25     | 42        | 0.896  | 0.786 | 0.955  | 0.887 |
| Su34         | 48     | 62        | 0.937  | 0.871 | 0.953  | 0.907 |
| Su57         | 41     | 72        | 0.971  | 0.929 | 0.979  | 0.945 |
| Tu160        | 40     | 54        | 0.959  | 0.926 | 0.97   | 0.921 |
| Tu95         | 26     | 36        | 0.965  | 0.758 | 0.91   | 0.877 |
| Tornado      | 43     | 63        | 0.961  | 0.774 | 0.924  | 0.882 |
| U2           | 38     | 44        | 0.976  | 0.934 | 0.987  | 0.956 |
| US2          | 84     | 90        | 0.981  | 0.944 | 0.981  | 0.942 |
| V22          | 68     | 100       | 0.988  | 0.847 | 0.947  | 0.853 |
| XB70         | 20     | 20        | 0.98   | 0.9   | 0.938  | 0.88  |
| YF23         | 13     | 18        | 0.968  | 0.833 | 0.953  | 0.941 |
| Vulcan       | 46     | 69        | 0.961  | 0.754 | 0.911  | 0.851 |
| J20          | 47     | 76        | 0.906  | 0.776 | 0.896  | 0.856 |

Globalement, les métriques montrent que le modèle a une bonne précision (0.944) et un bon rappel (0.86). Les valeurs de mAP50 et mAP50-95 indiquent une performance élevée pour la plupart des classes d'avions, avec des scores proches ou supérieurs à 0.9, ce qui démontre l'efficacité du modèle à détecter les avions dans les images. Il est important de noter que certaines classes ont des scores plus bas, ce qui peut être dû à des variations dans les données d'entraînement ou à des caractéristiques spécifiques des avions. Ces résultats peuvent être utilisés pour améliorer le modèle en ajustant les hyperparamètres ou en collectant davantage de données pour les classes sous-représentées.

## Test du modèle sur des images

### Test sur dataset test

Après l'entraînement et la validation du modèle, nous pouvons le tester sur des images réelles pour évaluer sa performance en conditions réelles. Voici un exemple de code pour tester le modèle sur quelques images du dataset test :

```python
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import glob
import random
import pandas as pd

# Charger le modèle YOLO
model = YOLO('/kaggle/input/yolov7-military-plane/yolov8-m-best.pt')

# Chemins vers les images et annotations de test
test_image_paths = sorted(glob.glob('/kaggle/working/ultralytics/data/test/images/*.jpg'))
test_annotation_paths = sorted(glob.glob('/kaggle/working/ultralytics/data/test/labels/*.txt'))

# Sélectionner quelques images aléatoires
sampled_indices = random.sample(range(len(test_image_paths)), 5)
sampled_image_paths = [test_image_paths[i] for i in sampled_indices]
sampled_annotation_paths = [test_annotation_paths[i] for i in sampled_indices]

# Faire des prédictions sur les images sélectionnées
results = model(sampled_image_paths)

for img_path, ann_path, result in zip(sampled_image_paths, sampled_annotation_paths, results):
    image = cv2.imread(img_path)[:, :, ::-1]
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()

    # Lire et afficher les annotations réelles
    with open(ann_path, 'r') as f:
        annotations = f.readlines()
    
    print(f"Annotations réelles pour {img_path}:")
    for annotation in annotations:
        class_num, x_center, y_center, b_width, b_height = map(float, annotation.split())
        class_name = model.names[int(class_num)]
        print(f"Classe: {class_name}")

    # Afficher les prédictions du modèle
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Déplacer les tenseurs vers la CPU et les convertir en numpy
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red')
        ax.add_patch(rect)
        plt.text(x1, y1, model.names[int(box.cls[0])], color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.show()
```

Ce code charge le modèle YOLOv8 entraîné, sélectionne quelques images aléatoires du dataset test, fait des prédictions sur ces images et affiche les résultats. Les annotations réelles sont également affichées dans des prints pour comparer les prédictions du modèle avec les vérités terrain.

### Test sur images réelles

En plus des images du dataset test, nous pouvons également tester le modèle sur des images hors échantillon pour évaluer sa capacité à généraliser à de nouvelles données. J'ai choisi une image de Rafale sur google image pour tester le modèle.

Voici l'image de Rafale utilisée pour le test et la prédiction du modèle :

![Rafale](/assets/img/AirCraft/RafalePred.jpg)

On peut voir que le modèle a correctement détecté et classifié les deux avions Rafale dans l'image, avec des boîtes englobantes précises et des prédictions de classe correctes. 

