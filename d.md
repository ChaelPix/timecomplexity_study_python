# Complexité Théorique des Algorithmes de Tri avec Justifications Mathématiques

Dans cette section, nous présentons une analyse détaillée des complexités théoriques de divers algorithmes de tri. Pour chaque algorithme, nous fournissons :

- Une brève description du fonctionnement de l'algorithme.
- Une analyse des complexités dans le meilleur cas, le cas moyen, et le pire cas, avec des justifications mathématiques.

Nous concluons avec un tableau récapitulatif des complexités.

---

## 1. Bubble Sort

### Description

Le **Bubble Sort** parcourt la liste à plusieurs reprises, compare les éléments adjacents, et les échange s'ils ne sont pas dans le bon ordre. Ce processus est répété jusqu'à ce que la liste soit triée.

### Analyse de la Complexité

#### Meilleur Cas : $O(n)$

- **Condition** : La liste est déjà triée.
- **Explication** :
  - L'algorithme effectue un passage pour vérifier si la liste est triée.
  - Avec une optimisation qui suit si des échanges ont été effectués, l'algorithme peut s'arrêter tôt s'il n'y a pas d'échanges.
  - **Nombre de Comparaisons** : $n - 1$
  - **Nombre d'Échanges** : 0
- **Justification** :
  $$
  T(n) = O(n)
  $$

#### Cas Moyen : $O(n^2)$

- **Condition** : Les éléments sont dans un ordre aléatoire.
- **Explication** :
  - En moyenne, les éléments doivent "remonter" à leur position correcte.
  - L'algorithme effectue environ $ \frac{n(n - 1)}{2} $ comparaisons et échanges.
- **Justification** :
  $$
  T(n) = O\left( \frac{n(n - 1)}{2} \right) = O(n^2)
  $$

#### Pire Cas : $O(n^2)$

- **Condition** : La liste est triée dans l'ordre inverse.
- **Explication** :
  - Chaque élément doit être comparé et échangé à chaque passage.
  - Le nombre maximum de comparaisons et d'échanges est atteint.
- **Justification** :
  $$
  T(n) = O\left( \frac{n(n - 1)}{2} \right) = O(n^2)
  $$

---

## 2. Selection Sort

### Description

Le **Selection Sort** divise la liste en une région triée et une région non triée. Il sélectionne à chaque itération l'élément minimum de la région non triée et l'échange avec le premier élément non trié.

### Analyse de la Complexité

#### Meilleur Cas : $O(n^2)$

- **Condition** : Quel que soit l'ordre initial.
- **Explication** :
  - Le Selection Sort effectue toujours le même nombre de comparaisons, quelle que soit l'ordre initial de la liste.
  - **Nombre de Comparaisons** : $ \frac{n(n - 1)}{2} $
  - **Nombre d'Échanges** : $n - 1$
- **Justification** :
  $$
  T(n) = O\left( \frac{n(n - 1)}{2} \right) = O(n^2)
  $$

#### Cas Moyen : $O(n^2)$

- **Condition** : Les éléments sont dans un ordre aléatoire.
- **Explication** :
  - Comme pour le meilleur cas, le nombre de comparaisons et d'échanges reste le même.
- **Justification** :
  $$
  T(n) = O(n^2)
  $$

#### Pire Cas : $O(n^2)$

- **Condition** : Quel que soit l'ordre initial.
- **Explication** :
  - La performance de l'algorithme n'est pas affectée par l'arrangement initial des éléments.
- **Justification** :
  $$
  T(n) = O(n^2)
  $$

---

## 3. Insertion Sort

### Description

L'**Insertion Sort** construit la liste triée finale un élément à la fois. Il prend chaque élément et l'insère à sa place dans la partie déjà triée de la liste.

### Analyse de la Complexité

#### Meilleur Cas : $O(n)$

- **Condition** : La liste est déjà triée.
- **Explication** :
  - Chaque élément est comparé une fois avec son prédécesseur.
  - Aucun décalage n'est nécessaire.
  - **Nombre de Comparaisons** : $n - 1$
- **Justification** :
  $$
  T(n) = O(n)
  $$

#### Cas Moyen : $O(n^2)$

- **Condition** : Les éléments sont dans un ordre aléatoire.
- **Explication** :
  - En moyenne, chaque nouvel élément est comparé à la moitié des éléments déjà triés.
  - **Comparaisons moyennes par insertion** : $ \frac{i}{2} $ pour l'élément $i^{e}$.
  - **Total des Comparaisons** :
    $$
    T(n) = \sum_{i=1}^{n} \frac{i}{2} = \frac{1}{2} \cdot \frac{n(n + 1)}{2} = O(n^2)
    $$
- **Justification** :
  $$
  T(n) = O(n^2)
  $$

#### Pire Cas : $O(n^2)$

- **Condition** : La liste est triée dans l'ordre inverse.
- **Explication** :
  - Chaque élément doit être comparé à tous les éléments précédents et décalé au début.
  - **Nombre de Comparaisons** : $ \frac{n(n - 1)}{2} $
- **Justification** :
  $$
  T(n) = O(n^2)
  $$

---

## 4. Heap Sort

### Description

Le **Heap Sort** convertit la liste en un tas binaire (max-tas ou min-tas). Il enlève ensuite de manière répétée la racine du tas (l'élément le plus grand ou le plus petit) et reconstruit le tas jusqu'à ce que tous les éléments soient triés.

### Analyse de la Complexité

#### Meilleur Cas : $O(n \log n)$

- **Condition** : Quel que soit l'ordre initial.
- **Explication** :
  - La construction du tas prend $O(n)$.
  - Chacune des $n - 1$ suppressions nécessite $O(\log n)$ pour réorganiser le tas.
- **Justification** :
  $$
  T(n) = O(n) + O(n \log n) = O(n \log n)
  $$

#### Cas Moyen : $O(n \log n)$

- **Condition** : Les éléments sont dans un ordre aléatoire.
- **Explication** :
  - Identique au meilleur cas.
- **Justification** :
  $$
  T(n) = O(n \log n)
  $$

#### Pire Cas : $O(n \log n)$

- **Condition** : Quel que soit l'ordre initial.
- **Explication** :
  - La complexité temporelle reste la même, quelle que soit la disposition initiale.
- **Justification** :
  $$
  T(n) = O(n \log n)
  $$

---

## 5. Quick Sort

### Description

Le **Quick Sort** est un algorithme de division et conquête. Il sélectionne un élément pivot et partitionne le tableau en deux sous-tableaux : les éléments inférieurs au pivot et les éléments supérieurs au pivot. Il trie ensuite récursivement les sous-tableaux.

### Analyse de la Complexité

#### Meilleur Cas : $O(n \log n)$

- **Condition** : Le pivot divise le tableau en deux moitiés égales à chaque étape.
- **Explication** :
  - La profondeur de la récursion est $\log n$.
  - À chaque niveau, le nombre total d'opérations est $n$.
- **Justification** :
  $$
  T(n) = 2T\left( \frac{n}{2} \right) + cn = O(n \log n)
  $$

#### Cas Moyen : $O(n \log n)$

- **Condition** : Les éléments sont dans un ordre aléatoire.
- **Explication** :
  - En moyenne, les partitions sont raisonnablement équilibrées.
  - En utilisant une analyse probabiliste, le nombre attendu de comparaisons est $ \approx 1.39 n \log n $.
- **Justification** :
  $$
  T(n) = O(n \log n)
  $$

#### Pire Cas : $O(n^2)$

- **Condition** : Le pivot est toujours le plus petit ou le plus grand élément.
- **Explication** :
  - La profondeur de la récursion devient $n$.
  - Le partitionnement ne réduit la taille du tableau que d'un seul élément à chaque fois.
- **Justification** :
  $$
  T(n) = T(n - 1) + cn = O(n^2)
  $$

---

## 6. Merge Sort

### Description

Le **Merge Sort** divise la liste en deux moitiés, trie récursivement chaque moitié, puis fusionne les moitiés triées pour produire une liste triée.

### Analyse de la Complexité

#### Meilleur Cas : $O(n \log n)$

- **Condition** : Quel que soit l'ordre initial.
- **Explication** :
  - La liste est toujours divisée en moitiés.
  - Fusionner deux listes triées prend $O(n)$.
- **Justification** :
  $$
  T(n) = 2T\left( \frac{n}{2} \right) + cn = O(n \log n)
  $$

#### Cas Moyen : $O(n \log n)$

- **Condition** : Les éléments sont dans un ordre aléatoire.
- **Explication** :
  - La performance de l'algorithme est indépendante de l'ordre initial.
- **Justification** :
  $$
  T(n) = O(n \log n)
  $$

#### Pire Cas : $O(n \log n)$

- **Condition** : Quel que soit l'ordre initial.
- **Explication** :
  - Identique aux cas meilleur et moyen.
- **Justification** :
  $$
  T(n) = O(n \log n)
  $$

---

## Conclusion

Les complexités temporelles des algorithmes de tri sont résumées dans le tableau suivant :

| Algorithme       | Meilleur Cas      | Cas Moyen         | Pire Cas          |
|------------------|-------------------|-------------------|-------------------|
| **Bubble Sort**     | $O(n)$            | $O(n^2)$          | $O(n^2)$          |
| **Selection Sort**  | $O(n^2)$          | $O(n^2)$          | $O(n^2)$          |
| **Insertion Sort**  | $O(n)$            | $O(n^2)$          | $O(n^2)$          |
| **Heap Sort**       | $O(n \log n)$     | $O(n \log n)$     | $O(n \log n)$     |
| **Quick Sort**      | $O(n \log n)$     | $O(n \log n)$     | $O(n^2)$          |
| **Merge Sort**      | $O(n \log n)$     | $O(n \log n)$     | $O(n \log n)$     |

---

**Remarque** : Les justifications mathématiques fournies incluent des analyses détaillées des scénarios de meilleur cas, cas moyen et pire cas pour chaque algorithme. Cela garantit une compréhension complète adaptée à un public de niveau ingénieur.
