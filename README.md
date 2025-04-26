# Analyse et Implémentation de Réseaux de Neurones Binarisés (BNN)

Ce dépôt contient le code source pour un projet INF8225 explorant l'analyse et l'implémentation de Réseaux de Neurones Profonds Binarisés (BNN) en utilisant PyTorch. L'objectif principal est d'étudier l'impact de la binarisation (quantification à 1 bit) sur les performances de différents modèles via l'approche QAT (Quantization-Aware Training).

## Sujet du Projet

La binarisation des réseaux de neurones est une technique de compression extrême visant à réduire drastiquement l'empreinte mémoire et le coût computationnel des modèles profonds, les rendant plus adaptés aux appareils à ressources limitées (Edge AI). Ce projet se concentre sur :

1.  L'implémentation de mécanismes de binarisation pour les poids et/ou activations en utilisant des couches PyTorch personnalisées et le Straight-Through Estimator (STE) pour l'entraînement (QAT).
2.  L'évaluation de l'impact de cette binarisation sur différentes architectures :
    * Perceptron Multi-Couches (MLP)
    * Architecture type VGG
    * MobileNetV2
    * Approche basée sur FracBNN/ResNet
3.  La conduite d'expériences sur des datasets standards : MNIST et CIFAR-10.
4.  L'analyse du compromis entre efficacité et précision résultant de la binarisation.

## Structure du Dépôt

* `/qnn`: Contient les modules principaux pour la quantification et les modèles (couches QAT, fonctions de binarisation, définitions des modèles).
* `/model_training`: Comprend les scripts pour la construction du modèle/dataset (`builder.py`), l'entraînement (`trainer.py`), et la gestion des résultats/graphiques.
* `/config`: Contient les fichiers de configuration (`.toml`) utilisés pour définir les expériences (modèle, dataset, hyperparamètres). Voir `config/readme.txt` pour plus de détails.
* `main.py`: Le script principal pour lancer un entraînement basé sur un fichier de configuration.
* `utils.py`: Fonctions utilitaires (ex: calcul de l'accuracy).

## Auteurs

* Roan Rubiales (Polytechnique Montréal)
* Quentin Luquet de Saint-Germain (Polytechnique Montréal)
