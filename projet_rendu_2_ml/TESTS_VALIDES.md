# Vérifications effectuées

## Compilation

Le projet a été compilé avec GCC/G++ et les options :

```text
-Wall -Wextra -Wpedantic
```

Les exécutables suivants ont été générés sans avertissement bloquant :

```text
demo
server
client
```

## Cas de test

```text
Données linéaires
Perceptron : 1.000
PMC        : 1.000
RBF        : 1.000
SVM        : 1.000

XOR
Perceptron : 0.500
PMC        : 1.000
RBF        : 1.000
SVM        : 0.500
```

## Sauvegarde / chargement

Les quatre modèles ont été sauvegardés puis rechargés. Leur accuracy est
restée identique après chargement.

## Client / serveur

Le serveur a chargé `models/perceptron_linear.txt`. Le client a envoyé la
donnée `(6, 6)` et a reçu la classe `1`.
