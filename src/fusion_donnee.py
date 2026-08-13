"""
Combine plusieurs nuits/bandes d'observation (deja traitees individuellement par pipeline()) en une seule carte Kp-Vsys plus
significative : somme les contributions temporelles brutes de chaque nuit avant de recalculer un unique z-score sur le total.
"""

from src import carte
from src import affichage as aff
from src import gestionnaire as g
import numpy as np



def fusionner_nuits(nuits, dossier):
    """
    Entrée :
        nuits   -> liste de tuples (CCF, tab_vitesses, tab_phase, berv, Vsys, tab_kp, tab_vsys, largeur_exclusion, Kp, element, poids_transit),
                   un par nuit/bande, exactement ceux renvoyés par pipeline
        dossier -> nom du sous-dossier de result/ où sauvegarder les figures
    """

    g.gestion_dossier_mixte(dossier)

    _, _, _, _, Vsys, tab_kp, tab_vsys, _, Kp, element, _ = nuits[0]
    largeur_exclusion = max(nuit[7] for nuit in nuits)  # le plus grand des largeur_exclusion calibres par bande

    n_kp, n_vsys = len(tab_kp), len(tab_vsys)
    somme_totale = np.zeros((n_kp, n_vsys))
    for CCF, tab_vitesses, tab_phase, berv, _, _, _, _, _, _, poids_transit in nuits:
        somme_totale += carte.carte_somme(CCF, Vsys, tab_vitesses, tab_phase, berv, tab_kp, tab_vsys, poids_transit)
    carte_z = carte.somme_to_zscore(somme_totale, tab_vsys, largeur_exclusion)
    aff.afficher_carte_kp_vsys(carte_z, tab_kp, tab_vsys, f"{dossier}/Carte/{element}", Kp)
    aff.afficher_tranche_kp(carte_z, tab_kp, tab_vsys, Kp, f"{dossier}/Tranche_carte/{element}")
