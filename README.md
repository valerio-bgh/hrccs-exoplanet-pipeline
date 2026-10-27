## Overview (English)
End-to-end Python pipeline for High-Resolution Cross-Correlation Spectroscopy (HRCCS): starting from reduced CRIRES+ spectra, it removes stellar and telluric systematics (PCA/SVD), cross-correlates the residuals with petitRADTRANS templates, and builds Kp–Vsys significance maps. Includes injection–recovery tests, automatic calibration of 8 free parameters, and abundance upper limits. Example data: public GJ 1214 b observations (Nortmann et al. 2026, CC-BY-4.0). Full documentation below is in French.

# Pipeline HRCCS — détection moléculaire dans les atmosphères d'exoplanètes en transit

Pipeline de spectroscopie à corrélation croisée haute résolution (HRCCS) pour détecter la présence d'une molécule
dans l'atmosphère d'une exoplanète en transit, à partir de données en .pickle au format d'un dictionnaire par
nuit/bande d'observation :
`{wave_modelA, spec_interpolated, err, snr, airmass, bjd_tdb, berv, tell}` (détail complet dans [Format des données d'entrée](#format-des-données-dentrée))

Chaque planète étudiée est décrite par un dictionnaire séparé de paramètres orbitaux/planétaires/stellaires
(période, éphéméride de transit, demi-grand axe, inclinaison, vitesse systémique, excentricité/argument du
périastre, rayon et masse de la planète, température d'équilibre, rayon/température effective/gravité de
surface/métallicité de l'étoile) à placer dans `src/simulation.py` (voir [Ajouter une planète](#ajouter-une-planète))

La méthode : PCA (SVD) pour retirer les systématiques temporelles, corrélation croisée avec un template pRT
(petitRADTRANS), et carte de significativité Kp-Vsys.

## Sommaire

- [Installation](#installation)
- [Structure du projet](#structure-du-projet)
- [Démarrage rapide](#démarrage-rapide)
- [Comment ça marche](#comment-ça-marche)
- [Le dictionnaire de configuration](#le-dictionnaire-de-configuration)
- [Ajouter une planète](#ajouter-une-planète)
- [Format des données d'entrée](#format-des-données-dentrée)
- [Résultats produits](#résultats-produits)
- [La calibration en détail](#la-calibration-en-détail)
- [Limite de détection (abondance)](#limite-de-détection-abondance)
- [Licence](#licence)

---

## Installation

Le projet utilise un environnement virtuel `venv/`, à créer après le clonage (`python -m venv venv`). Pour l'activer :

```bash
source venv/bin/activate
```

### Dépendances à installer

| Import Python | Paquet pip | Rôle |
|---|---|---|
| `numpy` | `numpy` | calcul tableau/matriciel, SVD |
| `scipy` | `scipy` | interpolation, filtres, optimisation |
| `matplotlib` | `matplotlib` | génération des figures (PDF dans `result/`) |
| `petitRADTRANS` | `petitRADTRANS` | génération des templates de spectre d'atmosphère (transfert radiatif) |
| `batman` | `batman-package` | modélisation de la courbe de transit (`traitement.transit`) |
| `ldtk` | `ldtk` | coefficients d'assombrissement centre-bord |

```bash
pip install numpy scipy matplotlib petitRADTRANS batman-package ldtk
```

Versions testées dans `venv/` : `numpy` 2.4.6, `scipy` 1.18.0, `matplotlib` 3.11.1, `petitRADTRANS` 3.3.3,
`batman-package` 2.5.2, `ldtk` 1.8.6.

Deux dépendances ont besoin de données externes téléchargées/mises en cache localement au premier usage
(connexion internet nécessaire la première fois) :
- **`petitRADTRANS`** : tables d'opacités, déjà téléchargées localement (`~/petitRADTRANS/input_data/...`) —
  premier chargement plus lent, ensuite mis en cache par petitRADTRANS lui-même.
- **`ldtk`** : spectres stellaires PHOENIX, téléchargés par `coefficients_assombrissement()` (via `LDPSetCreator`)
  au premier appel pour une nouvelle combinaison Tstar/loggstar/[Fe/H], mis en cache dans `~/.ldtk/`.

Les données d'observation vont dans `data/{nom_fichier}.pickle` (voir [Format des données d'entrée](#format-des-données-dentrée)).

### Cloner le projet

`venv/` et `ldtk/` sont exclus du dépôt (voir `.gitignore`) : propres à chaque machine, régénérables via
`pip install`. Après un `git clone`, il suffit donc de créer l'environnement virtuel et d'installer les
dépendances (ci-dessus) — rien d'autre à préparer.

**Données et résultats d'exemple** : une seule nuit, `data/220311_GJ1214.pickle`, ainsi que ses résultats
(`result/220311_GJ1214/`) et l'exemple de fusion multi-nuits (`result/GJ1214_b_fusionne/`, voir
[Fusionner plusieurs nuits/bandes](#fusionner-plusieurs-nuitsbandes)) sont inclus dans le dépôt à titre
d'exemple d'implémentation — le jeu de données complet (9 nuits) est téléchargeable sur https://zenodo.org/records/19387252
Ce sont des données de Nortmann et al. (2026), *« Cloudy with a chance of metals: Indications of CO2 in the
atmosphere of GJ 1214 b from high-resolution K-band spectroscopy »*, A&A — spectres CRIRES+ déjà réduits de
GJ 1214 b, publiés sur Zenodo sous licence CC-BY-4.0 : https://doi.org/10.5281/zenodo.19387252. Pour utiliser ses propres données, 
il suffit de déposer ses fichiers `.pickle` dans `data/` (voir [Format des données d'entrée](#format-des-données-dentrée)) ; 
les sous-dossiers de `result/` sont recréés automatiquement au premier appel de `pipeline()`, `test_injection()` ou `limite_abondance()` 
par `gestion_dossiers*()`/`ecrire_configuration()` de `src/gestionnaire.py`.

---

## Structure du projet

```
main.py               Point d'entrée principal : orchestre tout le traitement via pipeline()
test_injection.py     Variante de pipeline() avec injection d'un signal synthétique connu
limite_abondance.py   Limite supérieure sur l'abondance d'une molécule non détectée (injection-recovery + dichotomie)

src/
  calibration.py      Calibration automatique des paramètres de la pipeline
  fusion_donnee.py    Combine plusieurs nuits/bandes en une seule carte Kp-Vsys
  simulation.py       Dictionnaires planètes (paramètres orbitaux/planétaires/stellaires) + génération
                      du template de spectre (petitRADTRANS) + décalages Doppler + injection de signal
  traitement.py       Nettoyage, masquage, PCA (décomposition/reconstruction SVD), transit, propagation d'erreur
  carte.py            Corrélation croisée + construction de la carte de significativité Kp-Vsys (somme + z-score)
  gestionnaire.py     Ouverture des .pickle, création des dossiers de sortie, écriture de la config utilisée
  affichage.py        Génération de toutes les figures (PDF, dans result/)
```

** Les trois seuls fichiers destinés à être exécutés/appelés directement sont `main.py`, `test_injection.py` et
`limite_abondance.py` ** (à la racine — voir [Démarrage rapide](#démarrage-rapide) pour leur usage). 
Tous les autres fichiers `.py` du projet (`calibration.py`, `fusion_donnee.py`, `simulation.py`, `traitement.py`, `carte.py`, 
`gestionnaire.py`, `affichage.py`), sont présents dans `src/` et sont des modules de complément/support : ils fournissent 
des fonctions réutilisées par ces trois points d'entrée, mais ne sont pas conçus pour être lancés seuls.


---

## Démarrage rapide

```python
from src import simulation as s
from main import pipeline

# Détection complète sur un jeu de données, calibration automatique des paramètres
pipeline(s.GJ_1214_b, "220311_GJ1214", "H2O")

# Idem, mais affiche TOUTES les figures intermédiaires (spectres bruts/nettoyés/masqués, SNR,
# airmass, transit, évolution PCA...) en plus de la CCF et de la carte Kp-Vsys
pipeline(s.GJ_1214_b, "220311_GJ1214", "H2O", tout_afficher=True)

# Avec une configuration fixée à la main (évite la calibration, ~2 min économisées, mais moins fiable)
from main import configuration_defaut
pipeline(s.GJ_1214_b, "220311_GJ1214", "H2O", configuration=configuration_defaut)
```

Temps d'exécution indicatifs de `pipeline()`, hors calibration (~2 min, voir [La calibration en détail](#la-calibration-en-détail)) :
- `tout_afficher=False` (défaut) : quelques secondes, pour obtenir uniquement les cartes CCF, Kp-Vsys et la tranche
  de la carte Kp-Vsys au Kp théorique.
- `tout_afficher=True` : environ 1 min 30, pour obtenir en plus toutes les figures intermédiaires.
- Au total (calibration + `tout_afficher=True`) : environ 3 min 30.

### Fusionner plusieurs nuits/bandes

```python
from src.fusion_donnee import fusionner_nuits

nuit_1 = pipeline(s.GJ_1214_b, "220311_GJ1214", "H2O")   # nuit du 11/03/2022
nuit_2 = pipeline(s.GJ_1214_b, "220330_GJ1214", "H2O")   # nuit du 30/03/2022

fusionner_nuits([nuit_1, nuit_2], "GJ1214_b_merged")     # combine les deux, écrit dans result/GJ1214_b_merged/
```

`pipeline()` renvoie exactement le tuple attendu par `fusionner_nuits` — il suffit d'accumuler les résultats des
différentes nuits/bandes dans une liste et de la passer telle quelle.

### Tester la récupération d'un signal synthétique (injection-recovery)

```python
from test_injection import test_injection

test_injection(s.GJ_1214_b, "220311_GJ1214", "H2O", Kp_inject=100, Vsys_inject=15, amplitude_inject=1)
```

Injecte un signal planétaire connu (à Kp/Vsys/amplitude donnés) dans les données réelles, refait tout le
traitement, et vérifie si le pipeline retrouve bien ce signal à la bonne position. Utile pour valider que le
pipeline fonctionne correctement, indépendamment de la présence ou non d'un vrai signal dans les données.

### Calibrer sans lancer toute la détection

```python
from src.calibration import calibration

configuration = calibration(s.GJ_1214_b, "220311_GJ1214", "H2O")
# configuration est un dict réutilisable directement dans pipeline()/test_injection() via configuration=...
```

### Limite supérieure sur l'abondance (non-détection)

```python
from limite_abondance import limite_abondance

abondance_limite = limite_abondance(s.GJ_1214_b, "220311_GJ1214", "H2O", seuil_sigma=5.0)
```

Cherche par dichotomie (échelle log) la plus petite abondance qui, injectée au (Kp, Vsys) réel de la planète,
serait détectée à `seuil_sigma`. Voir [Limite de détection (abondance)](#limite-de-détection-abondance) pour la méthode.

---

## Comment ça marche

Pour un triplet (planète, fichier de données, molécule), `pipeline()` enchaîne :

1. **Nettoyage** : normalisation puis retrait des variations lentes de luminosité (spectre / médiane temporelle).
2. **Masquage** : bandes telluriques profondes, raies qui se décalent entre les deux prises A/B, valeurs NaN.
3. **PCA (SVD)** : retire les `r_max` premières composantes principales du cube masqué (systématiques
   temporelles communes à tous les pixels — tellurique résiduel, variations instrumentales).
4. **Génération du template** : spectre de transmission de la molécule via petitRADTRANS, dégradé à la
   résolution de l'instrument, élargissement rotationnel optionnel (profil de Gray), continuum potentiellement retiré.
5. **Corrélation croisée** : la CCF pondérée par l'erreur entre le résidu post-PCA et le template Doppler-décalé
   sur une grille de vitesses.
6. **Carte Kp-Vsys** : la CCF est alignée sur la courbe de vitesse radiale attendue pour chaque couple d'essai
   (Kp, Vsys), puis sommée sur le temps en pondérant chaque pose par `poids_transit` (profondeur de transit
   attendue, nulle hors transit) — ce qui exclut de fait les poses hors transit de la somme, celles-ci ne
   contenant aucun signal planétaire et ajoutant du bruit. Le résultat est ensuite converti en z-score par rapport 
   au bruit de fond (autres Vsys à Kp fixé, hors d'une fenêtre d'exclusion autour du point testé).

Un vrai signal planétaire se manifeste par un pic significatif dans cette carte, à l'endroit correspondant au
Kp et Vsys réels de la planète.

---

## Le dictionnaire de configuration

`pipeline()`, `test_injection()` et `limite_abondance()` acceptent tous un paramètre `configuration` (dict avec les
mêmes clés, optionnel et si non fourni, chacune appelle `calibration()` pour en obtenir un). 
`configuration_defaut` dans `main.py` est un exemple statique, utilisable pour sauter la calibration :

| Clé | Rôle |
|---|---|
| `tab_kp`, `tab_vsys` | Grilles d'écarts (km/s) autour du Kp/Vsys connus, testées dans la carte (réglé dans `src/calibration.py`) |
| `seuil_decallage` | Masque les pixels dont l'écart-type temporel dépasse ce facteur × l'écart-type médian de l'ordre (réglé dans `src/calibration.py`) |
| `propagation_erreurs` | Propage rigoureusement l'effet de la PCA sur l'erreur (au lieu de garder l'erreur brute masquée) (réglé dans `src/calibration.py`) |
| `elargissement_rotationnel` | Applique l'élargissement rotationnel (profil de Gray) au template (réglé dans `src/calibration.py`) |
| `abondance` | Fraction massique (mass_fraction) de la molécule recherchée dans l'atmosphère, utilisée par `spectre_transmission()` pour générer le template (réglé dans `src/calibration.py`) |
| `pourcentage_bas`, `pourcentage_haut`, `largeur_voisinage` | Masquage tellurique à échelle locale (avec cycle à hystérésis) — optimisé par `calibration()` |
| `r_max` | Nombre de composantes PCA retirées — optimisé par `calibration()` |
| `supprimer_continum`, `facteur_continum` | Retire (ou non) le continuum du template avant corrélation — optimisé par `calibration()` |
| `supprimer_mediane` | Retire la médiane de la CCF (par vitesse) avant de sommer sur le temps — optimisé par `calibration()` |
| `largeur_exclusion` | Demi-largeur (km/s) exclue autour du Vsys testé pour estimer le bruit de fond (z-score) — optimisé par `calibration()` |

`configuration_defaut` (dans `main.py`) permet de sauter la calibration pour aller plus vite, mais avec des
valeurs génériques non adaptées au jeu de données précis — à réserver aux tests rapides, pas à une détection
sérieuse.

---

## Ajouter une planète

Dans `src/simulation.py`, ajouter un dictionnaire sur le même modèle que `GJ_1214_b` :

```python
MA_PLANETE_b = {
    "name": "Ma Planète b",

    # --- Paramètres Orbitaux ---
    "P": ... ,           # Periode orbitale [secondes]
    "Tc": ... ,          # Éphéméride de transit [bjd_tdb_secondes]
    "a": ... ,           # Demi-grand axe [cm]
    "i": ... ,           # Inclinaison orbitale [degrés]
    "Vsys": ... ,        # Vitesse systémique [km/s]
    "ecc": ... , "w": ... ,
    "v_rot": None,       # calculé automatiquement juste après (rotation supposée synchrone)

    # --- Paramètres de la Planète ---
    "Rp": ... ,          # Rayon [cm]
    "Mp": ... ,          # Masse [g]  -- ATTENTION, impacte directement la profondeur du template (via g)
    "g": None,           # calculé automatiquement
    "Teq": ... ,         # Température d'équilibre [K]
    "Kp": None,          # calculé automatiquement

    # --- Paramètres de l'Étoile ---
    "Rstar": ... , "Tstar": (valeur, incertitude), "loggstar": (valeur, incertitude), "[Fe/H]": (valeur, incertitude),
}

MA_PLANETE_b["g"] = (cst.G * MA_PLANETE_b["Mp"]) / MA_PLANETE_b["Rp"]**2
MA_PLANETE_b["Kp"] = ((2*np.pi*MA_PLANETE_b["a"]/MA_PLANETE_b["P"]) * np.sin(np.radians(MA_PLANETE_b["i"]))) * 1e-5
MA_PLANETE_b["v_rot"] = (2*np.pi*MA_PLANETE_b["Rp"]/MA_PLANETE_b["P"]) * 1e-5
```

Tout est en cgs. `Tstar`/`loggstar`/`[Fe/H]` doivent être des tuples `(valeur, incertitude)` — utilisés par LDTk
pour estimer les coefficients d'assombrissement centre-bord (`coefficients_assombrissement()`).

Les molécules utilisables dépendent des tables d'opacités petitRADTRANS installées

---

## Format des données d'entrée

`data/{fichier}.pickle` doit contenir un dict avec les clés :

| Clé | Forme | Description |
|---|---|---|
| `wave_modelA` | (n_ordres, n_pixels) | longueurs d'onde \[nm\] |
| `spec_interpolated` | (n_ordres, n_temps, n_pixels) | spectres bruts |
| `err` | (n_ordres, n_temps, n_pixels) | erreurs associées |
| `snr` | (n_ordres, n_temps) | rapport signal/bruit |
| `airmass` | (n_temps,) | masse d'air à chaque pose |
| `bjd_tdb` | (n_temps,) | temps \[jours\] |
| `berv` | (n_temps,) | vitesse radiale barycentrique de la Terre \[m/s\] |
| `tell` | (n_ordres, n_pixels) | profondeur des raies telluriques (référence pour le masquage) |

---

## Résultats produits

Tout est écrit dans `result/{fichier}/` :

```
result/{fichier}/
  CCF/{element}.pdf                          carte de corrélation croisée
  Carte/{element}.pdf                        carte de significativité Kp-Vsys
  Tranche_carte/{element}.pdf                coupe de la carte au Kp théorique
  Evolution_PCA/{element}/...                évolution du spectre selon le nombre de composantes retirées
  Configurations/{element}.txt               configuration utilisée + paramètres d'appel (pour traçabilité)
  Injection/...                              mêmes sous-dossiers, pour les runs test_injection()
  Limite_abondance/
    Carte/{element}.pdf                      carte Kp-Vsys à l'abondance limite retenue
    Tranche_carte/{element}.pdf              coupe de cette carte au Kp théorique
    Zscore/{element}.pdf                     z-score au point injecté en fonction de l'abondance testée
    Configurations/{element}.txt             configuration utilisée + paramètres d'appel
```

`fusionner_nuits()` écrit dans `result/{dossier}/Carte/` et `result/{dossier}/Tranche_carte/` (dossier de sortie
choisi librement, indépendant des fichiers d'entrée fusionnés).

---

## La calibration en détail

Certains réglages (`TAB_KP`, `TAB_VSYS`, `SEUIL_DECALLAGE`, `PROPAGATION_ERREURS`, `ELARGISSEMENT_ROTATIONNEL`,
`ABONDANCE`...) ne sont pas optimisés automatiquement : ce sont des constantes fixes à ajuster directement en tête
du fichier `src/calibration.py`, dans sa section "Paramètres de pipeline" — voir aussi la colonne "réglé dans
`src/calibration.py`" du [tableau de configuration](#le-dictionnaire-de-configuration).

`calibration()` part d'une configuration de départ (`DEPART`) et ajuste successivement chaque paramètre en le
testant sur une grille de valeurs (`GRILLE_ESSAIS`), en injectant un signal synthétique connu à plusieurs
positions Kp-Vsys (`POSITIONS`) et en notant chaque combinaison sur 3 critères multipliés entre eux :

1. **A** — corrélation entre la carte obtenue et une gaussienne 2D centrée sur le point injecté
2. **retention** — amplitude du signal injecté récupérée après PCA / amplitude avant PCA
3. **localisation (d)** — écart entre le maximum réel de la carte et le point injecté, pénalisé exponentiellement

`r_max` est calibré deux fois (une fois en premier car c'est le paramètre le plus influent, une fois en dernier
une fois les autres paramètres fixés).

Avant tout ça, `choisir_amplitude()` détermine l'amplitude d'injection à utiliser : elle teste des amplitudes
croissantes (`AMPLITUDES_CALIBRATION`) et garde la première qui donne un score stable (positif) sur la
configuration de départ — une amplitude trop faible noie le signal injecté dans le bruit et rend la calibration
incapable de discriminer un bon réglage d'un mauvais.

Une calibration complète prend environ 2 minutes (dépend de la taille des données et du nombre d'ordres).

---

## Limite de détection (abondance)

`limite_abondance()` donne la plus petite abondance qui aurait été détectée à seuil_sigma près. En effet, une non-détection dans la vraie carte ne dit rien 
en soi sur l'abondance réelle, cette fonction quantifie la sensibilité de la pipeline pour donner une vraie limite supérieure exploitable.

Méthode (injection-recovery) :
1. Injecte un signal synthétique à une abondance candidate, **au (Kp, Vsys) réel** de la planète.
2. Refait tout le traitement (PCA, CCF, carte Kp-Vsys) et lit le z-score obtenu exactement à ce point.
3. Cherche par dichotomie, en échelle log sur l'abondance, la plus petite valeur qui donne un z-score ≥ `seuil_sigma`.

Le nettoyage/masquage/atmosphère pRT ne sont calculés qu'une fois (indépendants de l'abondance) ; seule la partie
template + injection + PCA + CCF est refaite à chaque abondance testée.

**Interet de la méthode et limites** :
- Utile pour rapporter une non-détection quantifiée et comparer la sensibilité entre molécules/bandes.
- Injecter puis corréler avec exactement le même template donne une sensibilité idéalisée, plus optimiste qu'une vraie recherche, attention à
  ne pas confondre avec une détection réelle.
- Une limite très supérieure aux abondances physiquement plausibles pour l'espèce (ex. quelques % en fraction
  massique) signifie que le jeu de données/la bande n'est simplement pas assez sensible à cette molécule, pas
  qu'elle est absente.
- *Si un pic (même faible, même non significatif) existe déjà au vrai Kp-Vsys dans les données réelles*, le signal
  injecté s'additionne à ce pic (l'injection est additive, la CCF est quasi-linéaire) : `abondance_limite` n'est alors
  plus la sensibilité pure du pipeline, mais l'abondance qu'il faudrait ajouter à ce qui est déjà présent pour
  que la somme franchisse `seuil_sigma`. Un moyen simple de vérifier si on est dans ce cas : regarder `z_min`, affiché
  au tout début de chaque run — comme `abondance_min` (1e-8 par défaut) injecte un signal négligeable, `z_min`
  représente essentiellement le z-score des données réelles seules. S'il est déjà notablement positif (2-3σ), la
  limite trouvée doit être interprétée comme un complément à ce qui est déjà là, pas comme une sensibilité "à partir
  de zéro".


---

## Licence

Le code de ce dépôt est publié sous licence MIT (voir `LICENSE`). Les données d'exemple (`data/220311_GJ1214.pickle`,
`result/`) proviennent de Nortmann et al. (2026), publiées sur Zenodo sous licence CC-BY-4.0 :
https://doi.org/10.5281/zenodo.19387252 — voir [Cloner le projet](#cloner-le-projet) pour le détail et l'attribution.
