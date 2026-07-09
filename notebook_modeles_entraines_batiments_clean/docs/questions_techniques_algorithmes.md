# Questions techniques sur les algorithmes

Cette section rassemble les informations qui peuvent être demandées pendant la présentation.  
L'objectif est de savoir précisément ce qui a été implémenté, sans devoir ouvrir tous les fichiers source.

### Vue d'ensemble

| Modèle | Type exact utilisé dans l'app | Multi-classe | Score utilisé pour choisir la classe |
|---|---|---|---|
| Perceptron | Perceptron linéaire de Rosenblatt | 3 modèles binaires en **un-contre-tous** | marge linéaire `w·x + b` |
| MLP / PMC | Réseau à une couche cachée | 3 MLP binaires en **un-contre-tous** | probabilité de chaque MLP |
| RBF | RBF naïf à centres fixes | 3 RBF binaires en **un-contre-tous** | probabilité de chaque RBF |
| SVM | SVM linéaire simple | 3 SVM binaires en **un-contre-tous** | marge linéaire `w·x + b` |

Le dataset a **3 classes** : Art déco, Art nouveau et Gothique.  
Comme les modèles de la librairie sont binaires, l'application crée donc **un modèle par classe** :

- modèle 0 : Art déco contre le reste ;
- modèle 1 : Art nouveau contre le reste ;
- modèle 2 : Gothique contre le reste.

À la prédiction, les trois modèles donnent chacun un score, puis l'application choisit la classe qui obtient le score le plus élevé.

---

### Détail du RBF

Le RBF utilisé ici est un **RBF naïf à centres fixes**.

Il n'utilise **pas** K-means.  
Il n'utilise **pas** l'algorithme de Lloyd.  
Les centres ne sont pas recalculés ou déplacés pendant l'entraînement.

Dans la librairie, les centres sont choisis directement parmi les exemples d'entraînement :

```cpp
for (int c = 0; c < numCenters; c++) {
    centers.row(c) = X.row(c % X.rows());
}
```

Cela signifie que les centres sont simplement pris dans les premières lignes du dataset d'entraînement, avec un retour au début si on demande plus de centres que d'exemples.

Pour le modèle RBF final du projet :

| Élément | Valeur |
|---|---:|
| Nombre de classes | 3 |
| Stratégie multi-classe | un-contre-tous |
| Nombre de RBF binaires | 3 |
| Centres par RBF | 16 |
| Centres stockés au total | 48 |
| Dimension d'un centre | 1024 |
| Sigma | 1 |
| Époques | 8 |
| Learning rate | 0.05 |

Attention à la nuance : il y a **16 centres par classifieur binaire**. Comme il y a 3 classes, l'application possède 3 RBF binaires. Les centres sont identiques au départ, car chaque RBF reçoit la même matrice `X`, mais chaque RBF apprend ses propres poids et son propre biais.

La fonction d'activation est gaussienne :

```text
activation = exp(-distance² / (2 × sigma²))
```

Une donnée proche d'un centre donne une activation forte.  
Une donnée loin du centre donne une activation faible.

Dans notre cas, le RBF fonctionne mal sur le dataset final : il prédit presque tout en **Gothique**. Ce résultat est cohérent avec une limite de ce réglage : seulement 16 centres fixes dans un espace de 1024 dimensions, avec `sigma = 1`, ce qui est très peu pour représenter des images de bâtiments.

---

### Détail du SVM

Le SVM utilisé est un **SVM linéaire**.

Il n'utilise pas de noyau polynomial.  
Il n'utilise pas de noyau RBF.  
Il n'utilise pas l'algorithme SMO complet.  
C'est une version pédagogique entraînée par mises à jour successives.

La décision se fait avec une marge linéaire :

```text
score = w·x + b
```

Pendant l'entraînement, on vérifie la marge :

```text
margin = label × score
```

- si `margin >= 1`, l'exemple est correctement placé avec une marge suffisante ;
- sinon, le modèle corrige les poids et le biais.

Pour le SVM final :

| Élément | Valeur |
|---|---:|
| Type | linéaire |
| Multi-classe | un-contre-tous |
| Nombre de SVM binaires | 3 |
| Époques | 12 |
| Learning rate | 0.005 |
| Lambda | 0.001 |
| Type de score | marge `w·x + b` |

---

### Détail du MLP / PMC

Le MLP est un réseau simple avec une couche cachée.

Dans cette application, il n'y a pas un seul réseau multi-classe avec trois sorties.  
Il y a **3 MLP binaires en un-contre-tous**.

Pour le MLP v2 retenu :

| Élément | Valeur |
|---|---:|
| Nombre de MLP binaires | 3 |
| Entrées | 1024 |
| Neurones cachés | 32 |
| Sortie | 1 probabilité par MLP binaire |
| Époques | 12 |
| Learning rate | 0.02 |
| Score utilisé | meilleure probabilité one-vs-rest |

Le MLP est le meilleur modèle obtenu dans cette version. Il reste simple, mais sa couche cachée lui permet de représenter des relations plus complexes que les modèles linéaires.

---

### Détail du Perceptron

Le perceptron est le modèle linéaire de base.

Pour chaque classe, il apprend un vecteur de poids et un biais.  
La correction suit la règle de Rosenblatt :

```text
poids = poids + learning_rate × vraie_réponse × exemple
biais = biais + learning_rate × vraie_réponse
```

Pour le perceptron final :

| Élément | Valeur |
|---|---:|
| Type | linéaire |
| Multi-classe | un-contre-tous |
| Nombre de perceptrons binaires | 3 |
| Époques | 12 |
| Learning rate | 0.01 |
| Type de score | marge `w·x + b` |

---

### Réponses rapides aux questions probables

| Question | Réponse courte |
|---|---|
| Le RBF utilise K-means ? | Non. |
| Le RBF utilise Lloyd ? | Non. |
| Les centres RBF bougent pendant l'entraînement ? | Non, ils sont fixes. |
| Comment les centres sont choisis ? | Ils sont pris directement dans les premières lignes du dataset d'entraînement. |
| Combien de centres RBF ? | 16 centres par RBF binaire, donc 48 centres au total pour 3 classes. |
| Quelle valeur de sigma ? | `sigma = 1`. |
| Pourquoi le RBF marche mal ? | Trop peu de centres fixes dans un espace image de 1024 dimensions ; les activations gaussiennes deviennent peu discriminantes. |
| Le SVM utilise un kernel ? | Non, c'est un SVM linéaire. |
| Le SVM est multi-classe directement ? | Non, il est utilisé en un-contre-tous. |
| Le MLP est multi-classe directement ? | Non, l'application utilise 3 MLP binaires en un-contre-tous. |
