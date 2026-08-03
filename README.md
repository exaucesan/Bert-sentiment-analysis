# Classification du type de violence avec BERT

## Le dataset

Le fichier `Train.csv` contient des tweets décrivant des expériences de violence, avec une colonne `type` qui indique la catégorie : `sexual_violence`, `Physical_violence`, `emotional_violence`, `economic_violence`, `Harmful_Traditional_practice`.

Le dataset est très déséquilibré : environ 32 600 exemples sont en `sexual_violence` contre seulement 188 pour `Harmful_Traditional_practice`. C'est le point le plus important à gérer dans ce projet.

## Choix d'entraînement

- Modèle : `bert-base-cased`, on prend le token `[CLS]` (`last_hidden_state[:,0]`) puis une couche linéaire vers 5 classes.
- Tokenisation : `max_length=128`, padding à taille max.
- Optimiseur : AdamW, `lr=2e-5`.
- 3 epochs, batch size 32, split train/val 90/10.
- Pour compenser le déséquilibre des classes, la loss (`CrossEntropyLoss`) est pondérée par classe (inverse de la fréquence). Sans ça le modèle a tendance à toujours prédire `sexual_violence`.
- Le suivi des métriques (loss/accuracy train et val) se fait avec Weights & Biases.

## Résultats

À compléter après l'entraînement :

| Epoch | Train loss | Train acc | Val loss | Val acc |
|-------|-----------|-----------|----------|---------|
| 1     |           |           |          |         |
| 2     |           |           |          |         |
| 3     |           |           |          |         |

## Difficultés rencontrées

- Le déséquilibre des classes rend l'accuracy globale trompeuse : un modèle qui prédit toujours la classe majoritaire aurait déjà ~82% d'accuracy. Il faudrait idéalement regarder aussi le F1-score par classe.
- Le texte des tweets contient des fautes, des emojis, de l'argot, ce qui rend la tâche plus difficile pour un tokenizer standard.
- Le temps d'entraînement de BERT sur ~36 000 exemples est assez long sans GPU.

## Lancer le projet

```bash
pip install -r requirements.txt
python modeling_bert.py   # entraîne le modèle et sauvegarde violence_model.pt
python demo.py            # lance la démo Gradio
```
