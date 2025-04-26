import torch
from torch import nn
from torch.nn import functional as F

from qnn.modules import QATLinear, QATConv2d, Binarize, clip  
from qnn.models.basic_models import ModelBase 

class InvertedResidual(nn.Module):
    """
    Implémente le bloc "Inverted Residual" de MobileNetV2, adapté pour 
    l'entraînement conscient de la quantification (Quantization-Aware Training - QAT).

    Ce bloc est la brique fondamentale de MobileNetV2 et se caractérise par :
    1.  Structure Inversée : Contrairement aux blocs résiduels classiques, 
        celui-ci commence par une expansion du nombre de canaux (étroit -> large)
        suivie d'une projection qui les réduit à nouveau (large -> étroit).
        
    2.  Convolution Séparable en Profondeur : Utilise une convolution 
        "depthwise" 3x3 (appliquée indépendamment sur chaque canal de la 
        représentation élargie) pour le filtrage spatial efficace.
        
    3.  Goulot d'Étranglement Linéaire (Linear Bottleneck) : La dernière 
        convolution de projection (1x1) est linéaire, c'est-à-dire qu'aucune 
        fonction d'activation (ReLU6) n'est appliquée après elle, afin de 
        préserver l'information dans la représentation étroite (bottleneck).
        
    4.  Connexion Résiduelle : Une connexion de raccourci (skip connection) 
        est ajoutée entre l'entrée et la sortie du bloc si les dimensions 
        spatiales et le nombre de canaux correspondent (stride=1 et inp=oup).
        
    5.  Adaptation QAT : Utilise les couches `QATConv2d`  à la place des 
        convolutions standard `nn.Conv2d` pour permettre l'entraînement et 
        l'inférence avec des poids potentiellement binarisés.

    Args:
        inp (int): Nombre de canaux d'entrée.
        oup (int): Nombre de canaux de sortie.
        stride (int): Stride (pas) pour la convolution depthwise (1 ou 2).
        expand_ratio (int): Facteur d'expansion pour la dimension cachée.
        dbits (int): Nombre de bits pour la quantification (utilisé par QATConv2d).
    """
    def __init__(self, inp, oup, stride, expand_ratio, dbits=1, *args, **kwargs):
        super(InvertedResidual, self).__init__()
        
        # --- Configuration initiale du bloc ---
        # Stride (pas) de la convolution depthwise (1 pour même dimension, 2 pour réduction)
        self.stride = stride 
        # Vérification que le stride est valide (MobileNetV2 utilise 1 ou 2)
        assert stride in [1, 2] 
        # Stocke le ratio d'expansion pour référence (utilisé pour la première conv et pour déterminer les canaux d'entrée de la depthwise)
        self.expand_ratio = expand_ratio 

        # --- Calcul des dimensions et conditions ---
        # Calcule la dimension cachée (nombre de canaux après expansion)
        hidden_dim = int(round(inp * expand_ratio)) 
        # Détermine si une connexion résiduelle doit être utilisée :
        # Condition : stride == 1 ET nombre de canaux d'entrée (inp) == nombre de canaux de sortie (oup)
        self.use_res_connect = self.stride == 1 and inp == oup 

        # Liste pour stocker les références aux tenseurs de poids des couches QAT (pour la fonction get_tensor_list)
        self.qat_weights_list = [] 

        # --- Définition des couches ---

        # 1. Couche d'Expansion (Pointwise Convolution 1x1)
        # Cette couche augmente le nombre de canaux si expand_ratio > 1
        if self.expand_ratio != 1:
            # Convolution 1x1 qui passe de 'inp' canaux à 'hidden_dim' canaux
            self.expand_conv = QATConv2d(inp, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False) # 
            # Normalisation par lot après la convolution d'expansion
            self.expand_bn = nn.BatchNorm2d(hidden_dim) 
            # Fonction d'activation ReLU6 après la normalisation
            self.expand_relu = nn.ReLU6(inplace=True) 
            # Ajoute le poids de cette couche QAT à la liste
            self.qat_weights_list.append(self.expand_conv.weight) 
        else:
            # Si expand_ratio == 1, il n'y a pas de couche d'expansion explicite.
            # Les couches sont mises à None pour indiquer qu'elles ne seront pas utilisées.
            # L'entrée de la couche depthwise sera directement 'inp'.
            self.expand_conv = None
            self.expand_bn = None
            self.expand_relu = None
            
        # 2. Couche Depthwise (Depthwise Convolution 3x3)
        # Applique une convolution spatiale indépendamment sur chaque canal.
        # Le nombre de canaux d'entrée dépend de si l'expansion a eu lieu ou non.
        dw_inp_channels = hidden_dim if self.expand_ratio != 1 else inp 
        # Convolution 3x3 Depthwise: 
        # - Nombre de canaux d'entrée = nombre de canaux de sortie = dw_inp_channels
        # - 'groups=dw_inp_channels' est ce qui la définit comme depthwise.
        # - 'stride' est appliqué ici pour potentiellement réduire la dimension spatiale.
        self.depthwise_conv = QATConv2d(dw_inp_channels, dw_inp_channels, kernel_size=3, stride=stride, padding=1, groups=dw_inp_channels, bias=False) # 
        # Normalisation par lot après la convolution depthwise
        self.depthwise_bn = nn.BatchNorm2d(dw_inp_channels) 
        # Fonction d'activation ReLU6 après la normalisation
        self.depthwise_relu = nn.ReLU6(inplace=True) 
        # Ajoute le poids de cette couche QAT à la liste
        self.qat_weights_list.append(self.depthwise_conv.weight) 

        # 3. Couche de Projection (Pointwise Convolution 1x1 - Linéaire)
        # Cette couche réduit le nombre de canaux pour correspondre à la sortie 'oup'.
        # Elle est "linéaire" car il n'y a PAS d'activation ReLU après la normalisation (Linear Bottleneck).
        # Convolution 1x1 qui passe de 'dw_inp_channels' (canaux après depthwise) à 'oup' (canaux de sortie finaux).
        self.project_conv = QATConv2d(dw_inp_channels, oup, kernel_size=1, stride=1, padding=0, bias=False) # 
        # Normalisation par lot après la convolution de projection
        self.project_bn = nn.BatchNorm2d(oup) 
        # PAS de ReLU6 ici !
        # Ajoute le poids de cette couche QAT à la liste
        self.qat_weights_list.append(self.project_conv.weight)


    def forward(self, x):
        # --- Sauvegarde de l'entrée pour la connexion résiduelle ---
        # Conserve une référence à l'entrée 'x' pour pouvoir l'ajouter à la fin 
        # si les conditions pour la connexion résiduelle sont remplies.
        identity = x 
        
        # --- 1. Phase d'Expansion (si expand_ratio > 1) ---
        # Si un facteur d'expansion est défini (différent de 1), applique la 
        # première convolution 1x1 (pour augmenter les canaux), suivie de 
        # BatchNorm et ReLU6.
        if self.expand_ratio != 1:
            # Applique Conv1x1 -> BN -> ReLU6
            out = self.expand_relu(self.expand_bn(self.expand_conv(x))) 
        else:
            # Si expand_ratio == 1, il n'y a pas d'expansion. 
            # La sortie de cette "phase" est simplement l'entrée 'x'.
            out = x 

        # --- 2. Phase Depthwise ---
        # Applique la convolution depthwise 3x3 (filtrage spatial), suivie de
        # BatchNorm et ReLU6. Le stride défini lors de l'initialisation 
        # (1 ou 2) est appliqué ici, ce qui peut réduire la taille spatiale.
        out = self.depthwise_relu(self.depthwise_bn(self.depthwise_conv(out)))

        # --- 3. Phase de Projection (Linéaire) ---
        # Applique la convolution de projection 1x1 (pour réduire les canaux), 
        # suivie de BatchNorm. 
        # Important : Il n'y a PAS de ReLU6 après cette étape (Linear Bottleneck).
        out = self.project_bn(self.project_conv(out)) 

        # --- 4. Connexion Résiduelle (si applicable) ---
        # Vérifie si la connexion résiduelle doit être utilisée (stride=1 et inp=oup).
        if self.use_res_connect:
            # Si oui, ajoute l'entrée sauvegardée ('identity') à la sortie calculée ('out').
            return identity + out 
        else:
            # Si non (stride=2 ou inp!=oup), retourne simplement la sortie calculée.
            return out

    def get_qat_weights(self):
        """ Helper to get weights from QATConv2d layers within this block """
        # Weights are already collected during init
        return self.qat_weights_list


class MobileNetV2(ModelBase): 
    """
    Implémentation du modèle MobileNetV2, adaptée pour QAT

    Caractéristiques Architecturales Clés :
    - Basé sur l'architecture MobileNetV2 (https://arxiv.org/abs/1801.04381).
    - Utilise des blocs "Inverted Residual" comme briques fondamentales.
    - Exploite les convolutions "Depthwise Separable" pour l'efficacité.
    - Intègre des "Linear Bottlenecks" pour préserver l'information.
    - Permet des connexions résiduelles entre les blocs.

    Args:
        num_classes (int): Nombre de classes de sortie pour le classifieur final.
                           Par défaut : 1000 (standard pour ImageNet).
        input_channels (int): Nombre de canaux de l'image d'entrée (e.g., 3 pour RGB).
                              Par défaut : 3.
        width_mult (float): Multiplicateur de largeur pour ajuster le nombre de 
                            canaux dans tout le réseau. Permet de créer des 
                            modèles plus petits ou plus grands. Par défaut : 1.0.
        dbits (int): Nombre de bits cible pour la quantification (passé aux 
                     couches QAT). Par défaut : 1 (binarisation).
        last_channel (int): Nombre de canaux dans la dernière couche 
                           convolutive avant le classifieur. Par défaut : 1280.

    Reference (Architecture Keras/TensorFlow): 
        https://github.com/keras-team/keras/blob/v3.3.3/keras/src/applications/mobilenet_v2.py
    """
    def __init__(self, num_classes=1000, input_channels=3, width_mult=1.0, dbits=1, last_channel=1280, *args, **kwargs):
        super(MobileNetV2, self).__init__() # 
        self.dbits = dbits
        
        # Configuration des blocs résiduels inversés pour MobileNetV2
        # Chaque sous-liste définit une étape/séquence de blocs :
        # [expansion_factor, num_output_channels, num_blocks, initial_stride]
        #   - expansion_factor (t) : Facteur d'expansion des canaux dans le bloc.
        #   - num_output_channels (c) : Nombre de canaux en sortie de cette étape.
        #   - num_blocks (n) : Nombre de blocs identiques dans cette étape.
        #   - initial_stride (s) : Stride (pas) pour le premier bloc de l'étape (les suivants ont un stride de 1).
        inverted_residual_setting = [
            # t,  c, n, s
            [1,  16, 1, 1],  # Bloc 0
            [6,  24, 2, 2],  # Blocs 1-2
            [6,  32, 3, 2],  # Blocs 3-5
            [6,  64, 4, 2],  # Blocs 6-9
            [6,  96, 3, 1],  # Blocs 10-12
            [6, 160, 3, 2],  # Blocs 13-15
            [6, 320, 1, 1],  # Bloc 16
        ]

        # --- Définition de la première couche ---
        current_input_channels = 32 # Nombre de canaux initial standard avant width_mult
        current_input_channels = int(current_input_channels * width_mult)
        self.conv1 = QATConv2d(
            in_channels=input_channels,          # <--- Doit être 3 pour CIFAR-10
            out_channels=current_input_channels, # <--- Doit être 32 (si width_mult=1.0)
            kernel_size=3, 
            stride=2, 
            padding=1, 
            bias=False
        ) 
        self.bn1 = nn.BatchNorm2d(current_input_channels)
        self.relu1 = nn.ReLU6(inplace=True)
        
        # --- Building inverted residual blocks ---
        self.features = nn.ModuleList()
        for expansion_factor, num_output_channels, num_blocks, initial_stride in inverted_residual_setting:
            # Appliquer le multiplicateur de largeur aux canaux de sortie
            current_output_channels = int(num_output_channels * width_mult) 
            
            # Créer les 'num_blocks' blocs pour cette étape
            for i in range(num_blocks):
                # Le stride initial est appliqué seulement au premier bloc de la séquence
                # Les blocs suivants ont un stride de 1
                stride = initial_stride if i == 0 else 1 
                
                # Ajouter le bloc InvertedResidual 
                self.features.append(
                    InvertedResidual( 
                        current_input_channels, 
                        current_output_channels, 
                        stride, 
                        expand_ratio=expansion_factor, 
                        dbits=dbits
                    )
                )
                
                # Le nombre de canaux d'entrée pour le bloc suivant est le nombre 
                # de canaux de sortie du bloc actuel
                current_input_channels = current_output_channels 
        
        # --- Building last several layers ---
        self.last_channel_width = int(last_channel * max(1.0, width_mult))
        self.conv_last = QATConv2d(current_input_channels, self.last_channel_width, kernel_size=1, stride=1, padding=0, bias=False) # 
        self.bn_last = nn.BatchNorm2d(self.last_channel_width)
        self.relu_last = nn.ReLU6(inplace=True)

        # --- Building classifier ---
        self.dropout = nn.Dropout(0.2)
        self.fc = QATLinear(self.last_channel_width, num_classes) 

        # --- Weight initialization ---
        self._initialize_weights()

    def _initialize_weights(self):
         for m in self.modules():
             if isinstance(m, (nn.Conv2d, QATConv2d)):  
                 nn.init.kaiming_normal_(m.weight, mode='fan_out')
                 if m.bias is not None:
                     nn.init.zeros_(m.bias)
             elif isinstance(m, nn.BatchNorm2d):
                 nn.init.ones_(m.weight)
                 nn.init.zeros_(m.bias)
             elif isinstance(m, (nn.Linear, QATLinear)):  
                 nn.init.normal_(m.weight, 0, 0.01)
                 if m.bias is not None:
                     nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Le tenseur d'entrée (batch d'images), 
                              généralement de forme (N, C_in, H_in, W_in).

        Returns:
            torch.Tensor: Le tenseur de sortie (logits ou prédictions brutes),
                          généralement de forme (N, num_classes).
        """
        
        # --- 1. Première Couche Convolutive ---
        # Applique la première convolution (QATConv2d), suivie de la 
        # normalisation par lot (BatchNorm) et de l'activation ReLU6.
        # Cette couche réduit généralement la dimension spatiale (stride=2).
        x = self.relu1(self.bn1(self.conv1(x)))
        
        # --- 2. Séquence de Blocs Inverted Residual ---
        # Itère séquentiellement à travers les blocs définis dans self.features (ModuleList).
        # Chaque bloc applique sa propre séquence d'expansion, depthwise conv, et projection.
        # La dimension spatiale peut être réduite davantage dans les blocs où stride=2.
        for block in self.features:
            x = block(x) # Appelle la méthode forward() de chaque InvertedResidualExplicit
            
        # --- 3. Dernière Couche Convolutive (avant pooling) ---
        # Applique la dernière convolution 1x1 (QATConv2d), suivie de 
        # BatchNorm et ReLU6. Prépare les caractéristiques pour le pooling.
        x = self.relu_last(self.bn_last(self.conv_last(x)))
        
        # --- 4. Pooling Global Moyen ---
        # Calcule la moyenne des caractéristiques sur les dimensions spatiales (Hauteur et Largeur).
        # Transforme le tenseur de (N, C_out, H, W) en (N, C_out).
        # C'est une alternative courante à nn.AdaptiveAvgPool2d(1) puis nn.Flatten().
        x = x.mean([2, 3]) # Calcule la moyenne sur les dimensions H (indice 2) et W (indice 3)
        
        # --- 5. Classifieur ---
        # Applique une couche Dropout pour la régularisation afin de réduire le surapprentissage.
        x = self.dropout(x)
        # Applique la couche linéaire finale (QATLinear) pour obtenir les scores (logits) 
        # pour chaque classe.
        x = self.fc(x)
        
        # Retourne les logits finaux
        return x
    
    def get_tensor_list(self):
        """
        Récupère et retourne une liste contenant les tenseurs de poids 
        (weight tensors) de toutes les couches QAT (QATConv2d, QATLinear) 
        présentes dans ce modèle MobileNetV2.

        Cette méthode est nécessaire pour les opérations spécifiques 
        liées à QAT ou au "freezing" de poids, où l'on a 
        besoin d'accéder directement aux paramètres qui seront quantifiés ou freeze. 
        
        Returns:
            list[torch.Tensor]: Une liste de tous les tenseurs de poids des 
                                couches QATConv2d et QATLinear du modèle.
        """
        # Initialise une liste vide pour collecter les tenseurs de poids
        tensor_list = [] 

        # 1. Récupérer le poids de la première couche QATConv2d 
        # On suppose que self.conv1 est QATConv2d et on accède directement à .weight
        tensor_list.append(self.conv1.weight)

        # 2. Récupérer les poids des blocs de caractéristiques (supposés InvertedResidualExplicit)
        # Itère à travers chaque bloc dans la ModuleList self.features
        for block in self.features:
            # On suppose que chaque 'block' a la méthode 'get_qat_weights'
            # et on l'appelle directement pour étendre la liste.
            tensor_list.extend(block.get_qat_weights())

        # 3. Récupérer le poids de la dernière couche QATConv2d 
        # On suppose que self.conv_last est QATConv2d et on accède directement à .weight
        tensor_list.append(self.conv_last.weight)

        # 4. Récupérer le poids de la couche QATLinear du classifieur 
        # On suppose que self.fc est QATLinear et on accède directement à .weight
        tensor_list.append(self.fc.weight)

        # Retourne la liste complète des tenseurs de poids QAT collectés
        return tensor_list