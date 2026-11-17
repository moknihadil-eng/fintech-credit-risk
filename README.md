# Credit Risk Analytics

Projet de scoring du risque de defaut Home Credit : exploration, preparation des donnees, comparaison des modeles et application Streamlit.

## Organisation

Navigation : [guide des notebooks](notebooks/README.md), [comparatif historique des modeles](reports/model_comparison.md), [rapport de parite](reports/preprocessing_parity.json).

Les fichiers de reference non utilises par le code sont ranges dans `docs/reference/`. Les documents generiques et extractions de travail sont archives dans `docs/archive/` et `reports/archive/`. Les deplacements sont traces dans `reports/organization_manifest.json`. Les classeurs encore references dans les notebooks restent a leur emplacement pour conserver leur fonctionnement.

- `data/raw/` : sources originales.
- `data/selected/` : selections intermediaires.
- `data/processed/` : tables preparees, split de validation et evaluations.
- `data/models/` : modele, schema des variables et metadonnees.
- `notebooks/` : analyses exploratoires, entrainements et interpretation.
- `src/` : scripts executables de traitement et de reentrainement.
- `app/` : application, pages et chargement commun des predictions.
- `docs/` : dictionnaires et documents d'analyse.

## Lancement

Depuis la racine du projet, dans un environnement disposant des dependances :

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

Les artefacts pickle existants doivent etre charges avec des versions compatibles avec celles de leur entrainement. Le fichier requirements.txt liste les dependances directes ; il ne constitue pas un verrouillage de versions.

`requirements-tested.txt` conserve les versions des dependances directes observees dans `.venv-1`, utilise pour les verifications de l'application. Il ne garantit pas la reproduction d'un environnement historique complet.

## Parcours de l'application

1. Vue d'ensemble : effectifs, defauts observes, scores et importance SHAP enregistree.
2. Analyse client : dossier de validation, import complet de variables preparees ou CSV application brut transforme par le pipeline historique verifie, puis explication SHAP.
3. Performance : mesures recalculees, simulation du seuil et export des predictions de validation.

## Resultats enregistres

Le notebook 18 et les artefacts SHAP indiquent 240 variables, 246 008 dossiers d'entrainement et 61 503 dossiers de validation, soit 307 511 dossiers. Le CSV model_final_evaluation.csv enregistre une ROC-AUC de 0,783309 et un KS de 0,429822 au seuil de classification 0,68.

Ces chiffres de validation ne sont pas des resultats sur un jeu de test independant. L'application recalcule ses indicateurs depuis le modele charge. Les scores d'un modele pondere ne sont pas automatiquement des probabilites calibrees.

## Points a verifier avant une presentation finale

- Executer les notebooks de modelisation dans un environnement reproductible et conserver les versions des bibliotheques.
- Comparer les cinq familles de modeles sur le meme split et les memes metriques ; distinguer recherche d'hyperparametres, validation croisee et evaluation finale.
- Verifier que l'imputation, l'encodage et la selection sont ajustes uniquement sur l'entrainement.
- Auditer le script `src/retrain_without_gender.py` : sa presence ne prouve pas que les artefacts ont ete reentraines sans CODE_GENDER.
- Le pipeline brut utilise les six tables historiques agregees locales. Il ne reconstruit pas ces tables a partir de nouveaux fichiers transactionnels ; leur actualisation reste une etape distincte.
- Conserver un test independant pour toute affirmation de generalisation.

Les fichiers historiques sont conserves pour ne pas casser les chemins des notebooks.

## Pipeline historique reutilisable

`src/preprocessing.py` reconstruit les variables depuis un CSV application brut et les six tables agregees existantes. Les encodages, valeurs d'imputation et constantes sont recuperes depuis les lignes d'entrainement sauvegardees, sans recalcul sur un nouveau client. Une categorie inconnue ou une imputation non recuperable provoque un refus explicite.

```powershell
.\.venv-1\Scripts\python.exe src/export_preprocessing.py
```

Le script n'exporte `data/models/preprocessing.json` qu'apres comparaison des 240 variables sur toutes les lignes d'entrainement et de validation. Le rapport est conserve dans `reports/preprocessing_parity.json`. La tolerance numerique est de 1e-12.

Cette reconstruction reproduit le comportement historique, y compris les indicateurs devenus constants lors des executions des notebooks. Elle ne corrige pas retroactivement une eventuelle fuite de donnees dans les traitements initiaux : cela necessiterait un nouveau pipeline ajuste sur l'entrainement et un reentrainement du modele.

## Data availability

Raw and processed datasets are excluded from this repository. To run the application, restore the Home Credit source tables, the six historical aggregate CSVs and `data/processed/xgb_split.pkl` locally. Model artifacts and the frozen preprocessing specification are included. Historical evaluation values remain available in reports/model_comparison.csv.
