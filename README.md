# Classification du type de violence avec BERT

## Le dataset

Le fichier `Train.csv` contient des tweets décrivant des expériences de violence, avec une colonne `type` qui indique la catégorie : `sexual_violence`, `Physical_violence`, `emotional_violence`, `economic_violence`, `Harmful_Traditional_practice`.

Le dataset est très déséquilibré : environ 32 600 exemples sont en `sexual_violence` contre seulement 188 pour `Harmful_Traditional_practice`. C'est le point le plus important à gérer dans ce projet.

## Choix d'entraînement

- Modèle : `bert-base-cased`, on prend le token `[CLS]` (`last_hidden_state[:,0]`) puis une couche linéaire vers 5 classes.
- Tokenisation : `max_length=64` (les tweets sont courts, pas besoin de 128).
- Optimiseur : AdamW, `lr=2e-5`.
- Pour compenser le déséquilibre des classes, la loss (`CrossEntropyLoss`) est pondérée par classe (inverse de la fréquence). Sans ça le modèle a tendance à toujours prédire `sexual_violence`.
- Le suivi des métriques (loss/accuracy train et val) se fait avec Weights & Biases (mode offline dans mon cas).

**Entraînement effectivement réalisé :** faute de GPU et de temps, l'entraînement complet sur les ~36 000 tweets originaux aurait pris plusieurs heures sur CPU. J'ai donc entraîné sur un sous-échantillon équilibré de 1605 exemples (`Train_sample.csv`), en gardant intactes les classes minoritaires (`economic_violence`: 217, `Harmful_Traditional_practice`: 188) et en limitant les classes majoritaires à 400 exemples chacune. Configuration utilisée : `batch_size=4`, `1 epoch`, split train/val 85/15.

## Résultats

Entraînement sur `Train_sample.csv` (1605 exemples, 1 epoch, batch_size=4) :

| Epoch | Train loss | Train acc | Val loss | Val acc |
|-------|-----------|-----------|----------|---------|
| 1     | 0.3808    | 0.8615    | 0.0722   | 0.9750  |

La val accuracy (97.5%) est calculée sur 240 exemples de validation. Elle est haute, en partie parce que certaines classes ont des marqueurs textuels assez distinctifs (mots-clés explicites liés au type de violence décrit), ce qui facilite la tâche même avec peu de données et un seul epoch.

## Difficultés rencontrées

- Le déséquilibre des classes rend l'accuracy globale trompeuse : un modèle qui prédit toujours la classe majoritaire aurait déjà ~82% d'accuracy sur le dataset complet. La pondération de la loss aide, mais il faudrait idéalement aussi regarder le F1-score par classe pour confirmer que les classes rares sont bien apprises.
- Le texte des tweets contient des fautes, des emojis, de l'argot, ce qui rend la tâche plus difficile pour un tokenizer standard.
- Le temps d'entraînement de BERT sur CPU est un vrai frein : impossible d'entraîner sur les 36 000 exemples originaux dans le temps imparti, d'où le choix d'un sous-échantillon réduit et équilibré pour ce rendu. Une piste d'amélioration serait d'utiliser un GPU (Colab par exemple) pour entraîner sur le dataset complet avec plusieurs epochs.
- Premier lancement plus lent que prévu au tout début de l'epoch (le temps que PyTorch charge le modèle en mémoire), ce qui a pu donner une impression trompeuse de blocage alors que le calcul avançait normalement.

## Lancer le projet

```bash
pip install -r requirements.txt
python modeling_bert.py   # entraîne le modèle et sauvegarde violence_model.pt
python demo.py            # lance la démo Gradio
```
