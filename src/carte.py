"""
Construit la carte de significativite Kp-Vsys : correlation croisee ponderee entre residu PCA et template, alignement sur la
courbe de vitesse radiale attendue puis somme temporelle (carte_somme), et conversion en z-score par rapport au bruit de fond.
"""

import warnings
import numpy as np

def correlation_croisee(spectre_residuel, err, template_3D, supprimer_mediane = False):
    """
    Calcule la matrice de corrélation croisée pondérée CCF(n, t, v)

    Entrée :
        spectrale_residuel -> (n_ordres, n_temps, n_pixels), résidus PCA
        err                -> (n_ordres, n_temps, n_pixels)
        template_3D        -> (n_vitesse, n_ordre, n_pixels)
        supprimer_mediane  -> (optionnel) supprime ou pas la médiane des vitesses pour chaque temps fixé

    Sortie :
        CCF -> (n_temps, n_v)
    """
    # On calcule CCF(n, t, v) = somme sur les pixels k de  residu[n,t,k] * template[v,n,k] / err[n,t,k]^2
    # C'est une somme de produits terme à terme sur k, donc un produit matriciel 
    poids = spectre_residuel / (err ** 2) # 1/err²

    poids = np.where(np.isfinite(poids), poids, 0.0)                 # (n_ordre, n_temps, n_pixel)
    template = np.where(np.isfinite(template_3D), template_3D, 0.0)  # (n_vitesse, n_ordre, n_pixel)


    template_reordonne = np.transpose(template, (1, 2, 0))           # (n_vitesse, n_ordre, n_pixel) -> (n_ordre, n_pixel, n_vitesse)

    # Pour chaque ordre n : (n_temps, n_pixel) @ (n_pixel, n_vitesse) -> (n_temps, n_vitesse)
    CCF = poids @ template_reordonne                                 # (n_ordre, n_temps, n_vitesse)

    CCF = CCF.sum(axis=0)                                        # (n_ordre, n_temps, n_vitesse) -> (n_temps, n_vitesse)
    
    if supprimer_mediane:
        CCF -= np.nanmedian(CCF, axis=1, keepdims=True)          # axe 1 = les vitesses

    return CCF



def carte_somme(CCF, Vsys, tab_vitesses, tab_phase, berv, tab_kp, tab_vsys, poids_transit=None):
    """
    Aligne la CCF sur la courbe de vitesse radiale attendue v(t) = Kp*sin(2*pi*phase_t) + Vsys - berv(t) et
    somme sur le temps, pour chaque (Kp, Vsys) de la grille d'essai

    Entrée :
        CCF          -> (n_temps, n_v), CCF déjà sommée sur les ordres
        Vsys         -> vitesse systémique connue de la planète (km/s)
        tab_vitesses -> (n_v,), grille de vitesses d'essai de la CCF (croissante)
        tab_phase    -> (n_temps,), phases orbitales
        berv         -> (n_temps,), vitesse radiale barycentrique de la Terre au moment de chaque pose (km/s)
        tab_kp, tab_vsys -> grilles d'essai (km/s)
        poids_transit -> (optionnel) (n_temps,), poids de chaque pose (typiquement (1-flux)/max(1-flux), cf. traitement.transit) :
                          pondère la somme temporelle par la profondeur de transit attendue, pour ne pas diluer le signal (nul hors
                          transit) avec des poses qui ne contiennent aucun signal planétaire. Défaut (None) = pas de pondération (= 1 partout)

    Sortie :
        carte -> (n_kp, n_vsys), carte des sommes brute alignée de la CCF pour chaque (Kp, Vsys)
    """
    n_temps = CCF.shape[0]
    n_kp, n_vsys = len(tab_kp), len(tab_vsys)
    carte = np.zeros((n_kp, n_vsys))

    if poids_transit is None:
        poids_transit = np.ones(n_temps)

    for ikp, kp in enumerate(tab_kp):
        vp_orbite = kp * np.sin(2 * np.pi * tab_phase) - berv       # (n_temps,),  Vp(kp,t) = Kp * sin(2*pi*phase_t) - berv(t)
        grille_vsys = Vsys + tab_vsys[None, :]
        vitesses_totales = vp_orbite[:, None] + grille_vsys   # (n_temps, n_vsys),  V(kp,t,vsys) = Vp(kp,t) + grille_vsys

        contributions = np.zeros((n_temps, n_vsys))
        for t in range(n_temps):
            contributions[t] = np.interp(vitesses_totales[t], tab_vitesses, CCF[t], left=np.nan, right=np.nan) * poids_transit[t]

        valide = np.isfinite(contributions)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            carte[ikp] = np.nansum(contributions, axis=0)
        carte[ikp][~valide.any(axis=0)] = np.nan

    return carte



def somme_to_zscore(carte, tab_vsys, largeur_exclusion):
    """
    Change l'echelle des points de la carte Kp-Vsys pour qu'ils representent un z-score = (x - mu_fond) / sigma_fond,
    où (mu_fond, sigma_fond) sont estimés sur le fond (tous les Vsys sauf une fenêtre d'exclusion autour du Vsys testé),
    indépendamment pour chaque Kp

    Entrée :
        carte          -> (n_kp, n_vsys), somme brute alignée pour chaque (Kp, Vsys), renvoyée par carte_somme
        tab_vsys          -> (n_vsys,), grille d'essai de Vsys (km/s)
        largeur_exclusion -> demi-largeur (km/s) exclue autour de chaque Vsys testé pour estimer le fond

    Sortie :
        carte_z -> (n_kp, n_vsys), carte en z-score
    """
    n_kp, n_vsys = carte.shape
    carte_z = np.zeros((n_kp, n_vsys))

    for ikp in range(n_kp):
        x_ligne = carte[ikp]
        for ivs, vsys in enumerate(tab_vsys):
            loin = np.abs(tab_vsys - vsys) > largeur_exclusion
            fond = x_ligne[loin]
            mu_fond = np.nanmean(fond)
            sigma_fond = np.nanstd(fond, ddof=1)
            # Suppprime les warning liées aux nan (optionnel)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                carte_z[ikp, ivs] = (x_ligne[ivs] - mu_fond) / sigma_fond # z-score = (z - moyenne)/equart-type

    return carte_z
