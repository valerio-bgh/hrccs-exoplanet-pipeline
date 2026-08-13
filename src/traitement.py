"""
Fonctions de traitement du signal : phases orbitales et transit (via batman), normalisation/nettoyage des spectres, 
masquage (bandes telluriques profondes, raies qui se decalent, nan), decomposition/reconstruction PCA (SVD) pour 
retirer les systematiques residuelles, et propagation rigoureuse de l'erreur a travers cette reconstruction PCA.
""" 

import warnings
import numpy as np
import batman
from scipy.ndimage import label, minimum_filter1d, maximum_filter1d


def phases_transit(bjd_tdb, P, Tc):
    """
    Convertit un tableau de temps (bjd_tdb) en phases orbitales et le centre au niveau de la transition

    Entrée :
        bjd_tdb -> tableau 1D (n_temps,) en secondes
        P       -> Periode orbitale en secondes
        Tc      -> éphéméride en secondes

    Sortie :
        phase -> tableau 1D (n_temps,), temps de chaque exposition, sans unité mais fraction d'une periode orbitale
    """
    phases = ((bjd_tdb - Tc) % P) / P        # toujours dans [0, 1[
    phases = np.where(phases >= 0.5, phases - 1.0, phases)   # recentre en [-0.5, 0.5[

    return phases



def transit(bjd_tdb, planete, u1, u2):
    """
    Indique, pour chaque temps d'observation, si la planète est en transit ou hors transit (baseline)

    Entrée :
        bjd_tdb -> tableau 1D (n_temps,) en secondes
        planete -> dictionnaire planète (défini dans src/simulation.py)
        u1, u2  -> coefficients d'assombrissement centre-bord quadratiques de l'étoile

    Sortie :
        poids_transit -> tableau 1D (n_temps,), poids normalisé du transit : (1-flux)/max(1-flux)
        en_transit    -> tableau booléen 1D (n_temps,), True = en transit, False = baseline
    """
    params = batman.TransitParams()
    params.t0 = planete["Tc"] + planete["P"] * round((bjd_tdb[0] - planete["Tc"]) / planete["P"])
    params.per = planete["P"]
    params.rp = planete["Rp"] / planete["Rstar"]
    params.a = planete["a"] / planete["Rstar"]
    params.inc = planete["i"]
    params.ecc = planete["ecc"]
    params.w = planete["w"]
    params.limb_dark = "quadratic"
    params.u = [u1, u2]

    modele = batman.TransitModel(params, bjd_tdb)
    flux = modele.light_curve(params)
    en_transit = flux < (1.0 - 1e-10)   # marge contre les imprecisions flottantes (hors transit le flux vaut théoriquement 1.0)
    poids_transit = (1 - flux) / np.max(1 - flux)

    return poids_transit, en_transit



# === Nettoyage ===

def normaliser_tableau_3D(data, denominateur = None):
    """
    Normalise chaque spectre à temps et ordre fixé à 1 ou à dénominateur

    Entrée : 
        data         -> de la forme (n_ordres, n_temps, n_pixels)
        denominateur -> 2D (ordre, temps,), valeur pour appliquer a "erreur" la même transformation

    Sortie :
        data_normaliser -> de la forme (n_ordres, n_temps, n_pixels)
    """
    if denominateur is None:
        # On enlève le Warning dû à l'ordre entièrement nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            max_vals = np.nanmax(data, axis=2, keepdims=True)
        # Si le maximum du nombre de photon reçu à ordre et temps fixé est nul, alors on le remplace par nan pour éviter la division pas 0
        max_vals[max_vals == 0] = np.nan
        denominateur = max_vals

    data_normaliser = data / denominateur

    return data_normaliser, denominateur



def nettoyage_tableau_3D(data, denominateur = None):
    """
    Réduit les différences de luminosité entre les différents spectres de meme ordre

    Entrée : 
        data         -> de la forme (n_ordres, n_temps, n_pixels)
        denominateur -> 2D (ordre, temps,), valeur pour appliquer a "erreur" la même transformation

    Sortie : 
        data_nettoye -> de la forme (n_ordres, n_temps, n_pixels)
    """
    if denominateur is None:
        # On enlève le Warning dû à l'ordre entièrement nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mediane = np.nanmedian(data, axis=2, keepdims=True)
        # Si la luminosite horizontale median est nul, alors on le remplace par nan pour éviter la division pas 0
        mediane[mediane == 0] = np.nan
        denominateur = mediane
    data_nettoye = data / denominateur
    
    return data_nettoye, denominateur



# === Masques ===

# Paramètres 1/n et 2/n et 3/n
def masque_tellurique(tell, pourcentage_bas = 30, pourcentage_haut=40, largeur_voisinage = 20):
    """
    Construit un masque 1D pour chaque ordre à partir d'une échelle locale : pour chaque pixel, le min et le max sont recalculés dans un voisinage de largeur 
    largeur_voisinage % de l'étendue totale en longueur d'onde de l'ordre (largeur_voisinage/2 % de part et d'autre du pixel), avant de normaliser ce pixel en 0-100% sur cette échelle locale
    Le seuillage ensuite est appliqué éventuellement avec cycle à hystérésis via pourcentage_haut

    Entrée :
        tell               -> tableau 2D (n_ordre, n_pixels)
        wavelength         -> tableau 2D (n_ordre, n_pixels), abscisse en longueur d'onde de chaque pixel
        pourcentage_bas    -> (optional) seuil bas en pourcentage (0 à 100), sur l'échelle locale
        pourcentage_haut   -> (optional) seuil haut en pourcentage (0 à 100), active le cycle à hystérésis si fourni 
        largeur_voisinage  -> (optional) largeur totale du voisinage local, en pourcentage de l'étendue en longueur d'onde de l'ordre (moitié de part et d'autre)

    Sortie :
        masque -> tableau booléen 2D (n_ordre, n_pixels), False = élément à masquer, True = élément à afficher
    """
    n_ordre, n_pixels = tell.shape
    masque = np.ones((n_ordre, n_pixels), dtype=bool)

    for n in range(n_ordre):
        if np.all(~np.isfinite(tell[n])):
            masque[n] = False

        else:
            demi_fenetre_pixels = max(1, int(round((largeur_voisinage / 2 / 100) * (n_pixels - 1))))
            taille_fenetre = 2 * demi_fenetre_pixels + 1

            tell_min_local = minimum_filter1d(tell[n], size=taille_fenetre, mode='nearest')
            tell_max_local = maximum_filter1d(tell[n], size=taille_fenetre, mode='nearest')

            denom = tell_max_local - tell_min_local
            denom = np.where(denom == 0, np.nan, denom)  # évite une division par 0 si le voisinage est plat
            echelle = ((tell[n] - tell_min_local) / denom) * 100

            if np.all(np.isnan(echelle)):
                masque[n] = False

            else:
                sous_seuil_bas = echelle < pourcentage_bas

                if pourcentage_haut is None:
                    masque[n] = ~sous_seuil_bas
                else:
                    sous_seuil_haut = echelle < pourcentage_haut

                    segments, n_segments = label(sous_seuil_haut) # Donne une etiquette a chaque groupe de TRUE consecutifs allant de 1 à n_segement et le reste 0 en revoie une liste d'entier et le nombre de groupe

                    segment_valide = np.zeros(n_segments + 1, dtype=bool)
                    labels_bas = segments[sous_seuil_bas]
                    labels_bas = labels_bas[labels_bas > 0]
                    segment_valide[labels_bas] = True # segments[sous_seuil_bas] contient une liste des numéro des indices comprenant un element un element sous seuil bas True, ensuite on marque dans segment_valide sur true à l'indice qui correspond au eleemnt de  segments[sous_seuil_bas]

                    a_masquer = segment_valide[segments]    # tous les groupe validé précedemment voient tout leur élements hériter du meme booléen

                    masque[n] = ~a_masquer

    return masque



def masque_decallage(data, seuil_decallage = 3.0):
    """
    Masque les pixels où une raie qui se décale au cours de la nuit (consequence d'une raie stellaire par exemple)
    Le critère est l'écart-type temporel de chaque pixel, normalisé par la médiane de cet écart-type sur l'ordre

    Entrée :
        data            -> tableau 3D (n_ordre, n_temps, n_pixel) (comme spectre_nettoyé ou err_nettoyée)
        seuil_decallage -> (optionel) un pixel est masqué si son écart-type dépasse ce facteur fois l'écart-type médian de son ordre

    Sortie :
        masque -> tableau booléen 2D (n_ordre, n_pixel), False = élément à masquer, True = élément à conserver
    """
    n_ordre, n_temps, n_pixel = data.shape
    masque = np.ones((n_ordre, n_pixel), dtype=bool)

    for n in range(n_ordre):
        # Ordre entièrement invalide (non calibré) : rien à conserver
        if np.all(~np.isfinite(data[n])):
            masque[n] = False
            continue

        # Supprime les warnings liés aux nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            variabilite = np.nanstd(data[n], axis=0)          # (n_pixel,), dispersion temporelle de chaque pixel
            reference = np.nanmedian(variabilite)             # niveau typique de l'ordre

        if not np.isfinite(reference) or reference == 0:      # si notre reference est nan ou nul, on marque l'ordre False
            masque[n] = False
            continue

        # Les pixels dont variabilite est nan sont aussi masqués, la comparaison renvoyant False
        masque[n] = (variabilite <= seuil_decallage * reference)

    return masque



def masque_nan(data):
    """
    Gère les nan d'une matrice revoyant un masque indiquant Flase si nan et True si vrai valeur

    Entrée : 
        data -> tableau de tailles quelconque (2D, 3D, 4D, ...)

    Sortie : 
        masque_binaire -> masque de meme dimension que data où True indique une vrai valeur, et False un nan
    """
    masque_binaire = ~np.isnan(data)
    return masque_binaire



def fusionner_masques(masque_1, masque_2):
    """
    Fusionne deux masques binaires 3D (ou 2D) de tailles compatibles : un pixel est True dans le masque de sortie seulement s'il est True dans les deux masques d'entrée
    Si l'un des deux masques a une dimension en moins (2D au lieu de 3D, ex: (n_ordre, n_pixel) au lieu de (n_ordre, n_temps, n_pixel)), il est automatiquement étendu en répétant ses valeurs le long de l'axe manquant (le temps), supposé être l'axe du milieu

    Entrée :
        masque_1 -> tableau booléen 3D (n_ordre, n_temps, n_pixel), ou 2D (n_ordre, n_pixel) s'il ne dépend pas du temps
        masque_2 -> tableau booléen, même forme que masque_1 en 3D, ou 2D (n_ordre, n_pixel) s'il ne dépend pas du temps

    Sortie :
        masque_fusionne -> tableau booléen 3D (n_ordre, n_temps, n_pixel), True seulement là où les deux masques d'entrée sont True
    """
    if masque_1.ndim == 2:
        masque_1 = masque_1[:, None, :]  # (n_ordre, 1, n_pixel), broadcast automatique sur le temps

    if masque_2.ndim == 2:
        masque_2 = masque_2[:, None, :]  # (n_ordre, 1, n_pixel), broadcast automatique sur le temps

    masque_fusionne = masque_1 & masque_2

    return masque_fusionne



def appliquer_masque_nul(data, masque):
    """
    Applique un masque où False = masquer = mettre 0, True = garder = converver valeur

    Entrée :
        data   -> tableau de dimension quelconque
        masque -> masque de meme dimension que data, False = pixel à mettre à 0, True = pixel à conserver

    Sortie :
        data_masque -> tableau 3D, même forme, 0 aux pixels masqués
    """
    data_masque = np.where(masque, data, 0)

    return data_masque



def appliquer_masque_nan(data, masque):
    """
    Réciproque de masque_nan : remet les nan dans une matrice aux emplacements indiqués par le masque binaire
    Entrée :
        data           -> tableau de taille quelconque (2D, 3D, 4D, ...) sans nan
        masque_binaire -> masque booléen de même forme que data où True = vraie valeur à conserver, False = emplacement où remettre un nan

    Sortie :
        data_nan -> data où les positions False du masque sont remises à nan
    """
    data_nan = np.where(masque, data, np.nan)

    return data_nan




# === PCA ===

def pca_3D_decompose(data):
    """
    Calcule la SVD pour chaque ordre d'un tableau 3D sans nan

    Entrée :
        data -> tableau 3D (n_ordres, n_temps, n_pixels) sans nan

    Sortie :
        U        -> (n_ordres, n_temps, k), k = min(n_temps, n_pixels)
        Sigma    -> (n_ordres, k), valeurs singulières de chaque ordre
        V_transp -> (n_ordres, k, n_pixels)
        w_barr   -> (n_ordres, 1, n_pixels), moyenne temporelle de chaque ordre
    """
    n_ordres, n_temps, n_pixels = data.shape
    k = min(n_temps, n_pixels)

    U = np.zeros((n_ordres, n_temps, k))
    Sigma = np.zeros((n_ordres, k))
    V_transp = np.zeros((n_ordres, k, n_pixels))
    w_barr = np.zeros((n_ordres, 1, n_pixels))

    for n in range(n_ordres):
        w_barr[n] = np.mean(data[n], axis=0, keepdims=True)
        B = data[n] - w_barr[n]

        U[n], Sigma[n], V_transp[n] = np.linalg.svd(B, full_matrices=False)

    return U, Sigma, V_transp, w_barr



def pca_3D_reconstruire(U, Sigma, V_transp, w_barr, r):
    """
    Reconstruit le cube 3D en retirant r composantes, à partir d'une décomposition déjà calculée 

    Entrée :
        U, Sigma, V_transp, w_barr -> decomposition de la SVD et signal moyen (résultat renvoyés par pca_3D_decompose)
        r                          -> nombre de composantes principales à retirer

    Sortie :
        data_pca -> tableau 3D (n_ordres, n_temps, n_pixels), r composantes retirées
    """
    n_ordres, n_temps, _ = U.shape
    n_pixels = V_transp.shape[2]
    data_pca = np.zeros((n_ordres, n_temps, n_pixels))

    for n in range(n_ordres):
        Sigma_tronque = Sigma[n].copy()
        Sigma_tronque[:r] = 0.0

        data_pca[n] = (U[n] * Sigma_tronque) @ V_transp[n]

        if r == 0:
            data_pca[n] += w_barr[n]

    return data_pca



def pca_3D_to_4D(U, Sigma, V_transp, w_barr, r_max):
    """
    Construit le tableau 4D (ordre, temps, pixel, r) à partir d'une décomposition déjà calculée 

    Entrée :
        U, Sigma, V_transp, w_barr, masques -> renvoyés par pca_3D_decompose
        r_max                               -> nombre max de composantes à retirer

    Sortie :
        data_4D -> tableau 4D (n_ordres, n_temps, n_pixels, r_max+1)
    """
    n_ordres, n_temps, _ = U.shape
    n_pixels = V_transp.shape[2]
    data_4D = np.zeros((n_ordres, n_temps, n_pixels, r_max + 1))

    for r in range(r_max + 1):
        data_4D[:, :, :, r] = pca_3D_reconstruire(U, Sigma, V_transp, w_barr, r)

    return data_4D



def propager_erreur(decomposition, err, r_max):
    """
    Propage rigoureusement l'erreur à travers la reconstruction PCA, en utilisant la même décomposition SVD que pca_3D_reconstruire.
    Pour chaque ordre, retirer les r_max premières composantes revient à appliquer à chaque pixel (suivi dans le temps) une
    matrice de projection fixe Q_r sur le sous-espace orthogonal aux composantes retirées :
            Q_r = I - U_r @ U_r^T           (U_r = les r_max premières colonnes de U)
    En propageant la variance à travers cette transformation linéaire (Q_r au carré TERME A TERME, pas une puissance matricielle) :
            variance_pca[j,k] = somme_i (Q_r[j,i])^2 * err[i,k]^2
            err_pca[j,k]      = sqrt(variance_pca[j,k])

    Entrée :
        decomposition -> tuple (U, Sigma, V_transp, w_barr, masque), U/Sigma/V_transp/w_barr renvoyés par pca_3D_decompose, masque -> masque binaire 3D (n_ordres, n_temps, n_pixels) utilisé pour la PCA
        err           -> tableau 3D (n_ordres, n_temps, n_pixels), erreur d'origine (même ordre que la decomposition)
        r_max         -> nombre de composantes retirées

    Sortie :
        err_pca -> tableau 3D (n_ordres, n_temps, n_pixels), erreur propagée (nan remis aux emplacements masqués, via appliquer_masque_nan)
    """
    U, Sigma, V_transp, w_barr, masque = decomposition
    n_ordres, n_temps, _ = U.shape
    n_pixels = V_transp.shape[2]
    err_pca = np.zeros((n_ordres, n_temps, n_pixels))

    for n in range(n_ordres):
        U_r = U[n, :, :r_max]
        Q_r = np.eye(n_temps) - U_r @ U_r.T

        variance_pca = (Q_r ** 2) @ (err[n] ** 2)   # propagation rigoureuse de la variance
        err_pca[n] = np.sqrt(variance_pca)

    return appliquer_masque_nan(err_pca, masque)
