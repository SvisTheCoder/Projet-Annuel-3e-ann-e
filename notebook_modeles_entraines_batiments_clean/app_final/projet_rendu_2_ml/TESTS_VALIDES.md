# Vérifications effectuées

## Compilation

Le projet a été compilé avec GCC/G++ et les options :

```text
-Wall -Wextra -Wpedantic
```

Les cibles principales sont :

```text
demo
train_cli
predict_cli
test_models
```

## Cas de test

```text
Données linéaires - accuracy sur des points jamais vus
Perceptron : 100,0 %
PMC        : 100,0 %
RBF        : 100,0 %
SVM        : 100,0 %

XOR bruité - accuracy sur des points jamais vus
Perceptron :  37,5 %
PMC        : 100,0 %
RBF        : 100,0 %
SVM        :  50,0 %

Trois classes simples - accuracy sur des points jamais vus
Perceptron : 100,0 %
PMC        : 100,0 %
RBF        : 100,0 %
SVM        : 100,0 %
```

Les exemples utilisés pour calculer l'accuracy de test sont maintenant séparés
des exemples d'entraînement. Les résultats à 100 % des cas linéaire et trois
classes restent possibles, car ces deux petits jeux artificiels sont
volontairement très faciles. Le XOR montre mieux la différence entre les
modèles linéaires et les modèles capables d'apprendre une frontière non
linéaire.

## Sauvegarde / chargement

Les quatre modèles ont été sauvegardés puis rechargés. Leur accuracy sur les
données de test est restée identique après chargement. Les fichiers temporaires
sont ensuite supprimés.
