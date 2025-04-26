Un fichier de configuration permet de personnaliser presque intégralement le(s) modèle(s) que l'on souhaite tester.

Une configuration représente un modèle unique si et seulement si aucun paramètre n'est une liste.
Si on souhaite faire varier des paramètres sans créer de nouvelles configurations, il est possible de remplacer
certains paramètres par une liste de paramètres à tester (c'est le cas de "models", "model_args", "freeze_agent", "freeze_args",
"epochs", "dataset", "batch_size"). L'environnement créera un produit cartésien des paramètres multiples et lancera toutes les 
sous-configurations générées. Les paramètres "model" et "model_args", anisi que "freeze_agent" et "freeze_args", fonctionnent 
en paire ((model1, model_args1), (model2, model_args2), etc.).

Les résultats sont générés automatiquement dans 'runs/', où les résultats sont stockés dans des .csv et les poids finaux dans des .pth.
On peut générer automatiquement des graphes avec le paramètre "plots" qui est une liste des types de graphes à faire.
Dans le cas où on utilise des paramètres multiples, on peut spécifier un paramètre pivot pour des graphes de comparaison: les sous-configurations
identiques (à l'exception du pivot) seront présentes dans le même graphe avec les valeurs possibles du pivot pour chaque courbe.