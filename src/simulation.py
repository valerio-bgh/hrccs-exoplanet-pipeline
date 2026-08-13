"""
Dictionnaires des parametres physiques (orbitaux, planetaires, stellaires) des systemes etudiés, et fonctions de simulation : 
generation du template de spectre de transmission via petitRADTRANS, mise a l'echelle instrumentale avec prise en compte possible de 
l'élargissement rotationnel, decalages Doppler, injection de signal planetaire synthetique dans les donnees
"""

from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS import physical_constants as cst
from scipy.ndimage import gaussian_filter1d
from ldtk import LDPSetCreator, BoxcarFilter
import numpy as np


GJ_1214_b = {
    "name": "GJ 1214 b",                   # jeu de données d'exemple : VLT/CRIRES+ K-band, Nortmann et al. 2026 (Zenodo 10.5281/zenodo.19387252, CC-BY-4.0)

    # --- Paramètres Orbitaux ---
    "P": 1.58040433 * 86400,               # Periode orbitale [jours] -> [secondes] (Cloutier et al. 2021)
    "Tc": 2455701.413328 * 86400.0,        # Éphemeride de transit [bjd_tdb] -> [bjd_tdb_secondes] (Cloutier et al. 2021)
    "a": 14.85 * (0.215 * cst.r_sun),      # a/Rs=14.85 (sans dimension) x Rstar [cm] (Cloutier et al. 2021)
    "i": 88.7,                             # Inclinaison orbitale [degres] (Cloutier et al. 2021)
    "Vsys": 21.1,                          # Vitesse du systeme [km/s] (Charbonneau et al. 2009, signe corrige par Kasper et al. 2020)
    "ecc": 0.0,                            # Excentricité (orbite circulaire, hypothese standard adoptee pour ce systeme)
    "w": 90.0,                             # Argument du periastre (sans effet comme ecc=0)
    "v_rot": None,                         # Rotation supposee synchrone (P_rot = P_orb) [km/s]

    # --- Paramètres de la Planète ---
    "Rp": 2.733 * cst.r_earth,             # Rayon de la planete [cm] (Mahajan et al. 2024)
    "Mp": 8.17 * cst.m_earth,              # Masse de la planete [g] (Cloutier et al. 2021)
    "g": None,                             # Gravité de surface [cm/s^2]
    "Teq": 567.0,                          # Temperature d'équilibre [K] (Mahajan et al. 2024)
    "Kp": None,                            # Semi-amplitude de la vitesse de la planète [km/s]

    # --- Paramètres de l'Étoile ---
    "Rstar": 0.215 * cst.r_sun,            # Rayon de l'etoile [cm] (Cloutier et al. 2021)
    "Tstar": (3250, 100),                  # (Teff, incertitude) [K] (Cloutier et al. 2021)
    "loggstar": (5.026, 0.040),            # (log g, incertitude) [dex, cm/s^2] (Cloutier et al. 2021)
    "[Fe/H]": (0.29, 0.12),                # (metallicité, incertitude) [dex] (Cloutier et al. 2021)
}

GJ_1214_b["g"] = ((cst.G * GJ_1214_b["Mp"]) / GJ_1214_b["Rp"]**2)
GJ_1214_b["Kp"] = ((2 * np.pi * GJ_1214_b["a"] / GJ_1214_b["P"]) * np.sin(np.radians(GJ_1214_b["i"]))) * 1e-5
GJ_1214_b["v_rot"] = (2 * np.pi * GJ_1214_b["Rp"] / GJ_1214_b["P"]) * 1e-5



def coefficients_assombrissement(planete, wave_modelA):
    """
    Estime les coefficients d'assombrissement centre-bord de l'étoile via LDTk, à partir des paramètres stellaires réels
    (Tstar, loggstar, [Fe/H]) et de la bande de longueurs d'onde observée. 
    Les coefficients sont :
        - quadratique (u1, u2)        -> utilisée par transit() pour la forme de la courbe de lumière
        - linéaire (epsilon_lineaire) -> utilisée par noyau_rotationnel()/spectre_pRT_to_CRIRES() pour l'élargissement rotationnel

    Entrée :
        planete     -> dictionnaire planète (défini dans src/simulation.py), doit contenir Tstar, loggstar et [Fe/H] en tuples (valeur, incertitude)
        wave_modelA -> tableau 2D (n_ordres, n_pixels) des longueurs d'onde (nm) de l'observation

    Sortie :
        u1, u2           -> float, coefficients d'assombrissement centre-bord quadratiques (loi de Kopal 1950)
        epsilon_lineaire -> float, coefficient d'assombrissement centre-bord linéaire (loi de Gray 1976)
    """
    valeurs_valides = wave_modelA[(wave_modelA != 0) & (~np.isnan(wave_modelA))]
    lambda_min = valeurs_valides.min()
    lambda_max = valeurs_valides.max()

    filtre = BoxcarFilter("bande", lambda_min, lambda_max)
    sc = LDPSetCreator(teff=planete["Tstar"], logg=planete["loggstar"], z=planete["[Fe/H]"], filters=[filtre], dataset="visir-lowres") # dataset="visir-lowres" : etendu jusqu'a 5500 nm -- couvre H ET K
    profils = sc.create_profiles()
    u1, u2 = profils.coeffs_qd(do_mc=False)[0][0] # coefficients quadratiques sans incertitudes
    epsilon_lineaire = float(profils.coeffs_ln(do_mc=False)[0][0][0])

    return u1, u2, epsilon_lineaire



def grille_kp(Kp, tab_kp):
    """
    Centre la grille sur Kp (la Kp connu noté dans le dicionnaire de la planète), en excluant les valeurs négatives

    Entrée :
        Kp     -> Kp connu autour duquel centrer le balayage (km/s)
        tab_kp -> grille d'écarts autour de 0 (km/s), typiquement symétrique en 0

    Sortie :
        tab_kp -> tableau 1D, grille de Kp d'essai absolue, toujours >= 0
    """
    tab_kp = tab_kp + Kp
    return tab_kp[tab_kp >= 0]



def bornes_wavelength(wavelength, vitesses_decallage):
    """
    Revoie les bones du tableau de longueur d'onde tout ordre confondu, hors 0, hors nan

    Entree : 
        wavelength -> tableau 2D (n_ordres, n_pixels) de longueurs d'ondes
        vitesses_decallage -> tableau 1D (n_vitesses,), grille de vitesses Doppler (km/s) couvrant, avec marge, toutes les combinaisons (Kp, Vsys, phase) possibles
    
    Sortie : 
        lambda_min -> float, longueur d'onde min
        lambda_max -> float, longueur d'onde min
    """
    valeurs_valides = wavelength[(wavelength != 0) & (~np.isnan(wavelength))]
    marge_rel = np.max(np.abs(vitesses_decallage)) / (cst.c * 1e-5)
    lambda_min = np.nanmin(valeurs_valides) * (1 - marge_rel) * 1e-3
    lambda_max = np.nanmax(valeurs_valides) * (1 + marge_rel) * 1e-3
    
    return lambda_min, lambda_max



def vitesse_decallage(phases, Vsys, tab_vsys, tab_kp, berv):
    """
    Calcule la grille de vitesses Doppler nécessaire pour couvrir toute la carte Kp-Vsys demandée, à partir de la plage de phase observée

    Entrée :
        phases   -> tableau 1D (n_temps,) des phases orbitales
        Vsys     -> vitesse systémique connue de la planète (km/s)
        tab_vsys -> tableau 1D (n_vsys,) des écarts de Vsys d'essai (km/s), autour de 0
        tab_kp   -> tableau 1D (n_kp,) des Kp d'essai (km/s)
        berv     -> tableau 1D (n_temps,), vitesse radiale barycentrique de la Terre à chaque pose (km/s)

    Sortie :
        vitesses_decallage -> tableau 1D (n_vitesses,), grille de vitesses Doppler (km/s) couvrant, avec marge, toutes les combinaisons (Kp, Vsys, phase, berv) possibles
    """
    sin_min = np.sin(2 * np.pi * phases).min()
    sin_max = np.sin(2 * np.pi * phases).max()

    tab_vsys_absolu = Vsys + tab_vsys

    v_min_necessaire = tab_vsys_absolu.min() + tab_kp.max() * sin_min - berv.max()
    v_max_necessaire = tab_vsys_absolu.max() + tab_kp.max() * sin_max - berv.min()

    marge = 5   # petite marge de sécurité en km/s
    v_min = int(np.floor(v_min_necessaire - marge))
    v_max = int(np.ceil(v_max_necessaire + marge))

    pas = np.mean(np.diff(tab_vsys)) # pas moyen de v_sys

    vitesses_decallage = np.arange(v_min, v_max + 1, pas)

    return vitesses_decallage



def creer_atmosphere(bornes_wavelength, element):
    """
    Initialise l'objet Radtrans (petitRADTRANS) pour la molécule recherchée, sur la plage de longueurs d'onde nécessaire

    Entrée :
        bornes_wavelength -> tuple (lambda_min, lambda_max) en microns
        element            -> (string) molécule à inclure comme "line_species" (dépend des tables d'opacités petitRADTRANS installées)

    Sortie :
        atmosphere -> objet Radtrans, réutilisable pour générer des spectres à différentes abondances (spectre_transmission)
    """
    atmosphere = Radtrans(
        pressures=np.logspace(-10, 2, 130),
        line_species=[element],
        rayleigh_species=["H2", "He"],
        gas_continuum_contributors=["H2--H2", "H2--He"],
        wavelength_boundaries=bornes_wavelength,   # en microns
        line_opacity_mode="lbl"
    )
    return atmosphere



def spectre_transmission(atmosphere, planete, element, abondance = 1e-3):
    """
    Génère le spectre de transmission (profondeur de transit en fonction de la longueur d'onde) de la molécule recherchée,
    via petitRADTRANS, à partir d'un profil isotherme (Teq) et d'une fraction massique constante en pression

    Entrée :
        atmosphere -> objet Radtrans (renvoyé par creer_atmosphere), déjà configuré pour la molécule et la plage de longueur d'onde
        planete    -> dictionnaire planète (défini dans src/simulation.py), utilise Teq, g, Rp, Rstar
        element    -> (string) molécule recherchée (doit correspondre au line_species de l'atmosphere)
        abondance  -> (optionnel) fraction massique (mass_fraction) de la molécule dans l'atmosphère

    Sortie :
        wave_template    -> tableau 1D, longueurs d'onde du template (nm)
        spectre_template -> tableau 1D, profondeur de transit -(Rp_lambda/Rstar)^2, sans unité
    """
    temperatures = planete["Teq"] * np.ones_like(atmosphere.pressures)


    mass_fractions = {
        'H2': 0.74 * np.ones_like(temperatures),
        'He': 0.24 * np.ones_like(temperatures),
        element: abondance * np.ones_like(temperatures),
    }
    mean_molar_mass = 2.33 * np.ones_like(temperatures)

    wavelengths, transit_radii, _ = atmosphere.calculate_transit_radii(
        temperatures=temperatures,
        mass_fractions=mass_fractions,
        mean_molar_masses=mean_molar_mass,
        reference_gravity=planete["g"],
        planet_radius=planete["Rp"],
        reference_pressure=1e-2,
    )

    wave_template = wavelengths * 1e7 # longeurs d'ondes en nanometre
    spectre_template = -(transit_radii / planete["Rstar"]) ** 2 # profondeur des raies sans unitées

    return wave_template, spectre_template



# wavelength de pRT a un pas de ln(lambda) et 1/R = dalta_lambda/lamda = delta(ln(lambda))
def spectre_pRT_to_CRIRES(wave_template, spectre_template, supprimer_continum = "True", facteur = 50, elargissement_rotationnel = False, v_rot = None, epsilon = 0.6, resolution = 100000):
    """
    Convolue le HRS avec une gaussienne pour ramener le template à la résolution de l'instrument (resolution), puis convolue avec une gaussienne large pour enlever le continuum
    (comme il est supprimé sur les données par la PCA). Applique éventuellement avant tout l'élargissement rotationnel du template (profil de Gray)

    Entrée :
        wave_template             -> (tableau 1D) longueurs d'onde (grille régulière en ln)
        spectre_template          -> (tableau 1D) profondeur de transit
        supprimer_continum        -> (optionnel, bool) retire ou non le continuum du template (deuxième convolution, plus large que celle de la résolution) avant de le renvoyer
        facteur                   -> (optionnel) facteur multiplicatif entre sigma_pixel (résolution) et la largeur de la convolution utilisée pour retirer le continuum
        elargissement_rotationnel -> (optionnel, bool) applique ou non l'élargissement rotationnel avant la convolution instrumentale
        v_rot                     -> (optionnel, km/s) vitesse de rotation équatoriale de la planète (bord du disque), requis si elargissement_rotationnel=True
        epsilon                   -> (optionnel) coefficient d'assombrissement centre-bord linéaire utilisé par noyau_rotationnel si elargissement_rotationnel=True
        resolution                -> (optionnel) résolution spectrale de l'instrument, utilisée pour dégrader le template à la résolution réelle des données (défaut 100000, CRIRES+)

    Sortie :
        spactre_arrange -> spectre (tableau 1D) arrangé
    """
    # Une convolution a sigma_pixel = enlever un ordre
    dln = np.mean(np.diff(np.log(wave_template)))  # =  pas réel de la grille ( ln(lambda) )

    if elargissement_rotationnel:  # convolution par le profil de Gray 
        noyau = noyau_rotationnel(v_rot, dln, epsilon)
        spectre_template = np.convolve(spectre_template, noyau, mode="same")

    FWHM_en_lnlambda = 1 / resolution
    FWHM_en_pixel = FWHM_en_lnlambda / dln
    sigma_pixel = FWHM_en_pixel / (2 * np.sqrt(2 * np.log(2))) # FWHM = 2 sqrt(2ln(2)) sigma

    spactre_arrange = gaussian_filter1d(spectre_template, sigma=sigma_pixel)

    if supprimer_continum:
        continuum = gaussian_filter1d(spactre_arrange, sigma=sigma_pixel * facteur) # Il faut un facteur entre le nombre de pixel par raie et le nombre de pixel par ordre
        spactre_arrange -=  continuum   # centré sur 0, ne garde que les raies

    return spactre_arrange



def noyau_rotationnel(v_rot, dln, epsilon):
    """
    Construit le noyau de convolution de l'élargissement rotationnel (profil de Gray, assombrissement centre-bord linéaire de coefficient epsilon),
    échantillonné sur la même grille régulière en ln(lambda) que le template (pas dln, 1 pixel = dln*c en km/s, approximation non relativiste)

    Entrée :
        v_rot   -> vitesse de rotation équatoriale à la limite du disque (km/s)
        dln     -> pas de la grille en ln(lambda) du template (renvoyé par spectre_pRT_to_CRIRES)
        epsilon -> coefficient d'assombrissement centre-bord linéaire

    Sortie :
        noyau -> tableau 1D, noyau de convolution discret normalisé (somme = 1)
    """
    pas_vitesse = dln * (cst.c * 1e-5)   # km/s par pixel

    n_half = max(1, int(np.ceil(v_rot / pas_vitesse)))
    v = np.arange(-n_half, n_half + 1) * pas_vitesse
    x2 = np.clip((v / v_rot) ** 2, 0, 1)   # clip pour eviter un sqrt negatif (warning) sur les points hors profil, mis a 0 par le where juste apres

    noyau = np.where(np.abs(v) < v_rot, 2 * (1 - epsilon) * np.sqrt(1 - x2) + 0.5 * np.pi * epsilon * (1 - x2), 0.0)
    noyau /= noyau.sum()   # renormalisation numerique de la version discretisee (la constante de normalisation analytique du profil continu ne s'applique pas telle quelle a la version discrete)

    return noyau



def doppler(wave_template, vitesses_decallage):
    """
    Génère un tableau 2D de longueur d'onde décallé pour chaque vitesse de la grille

    Entrée :
        wave_template      -> tableau 1D (n_pixels,) qui correspond aux longueurs d'onde au repos
        vitesses_decallage -> tableau 1D (n_vitesses, ) correspondant aux vitesses de shift

    Sortie :
        wave_decale -> tableau 2D (n_v, n_pixels) wave_decale[v, k] = wave_template[k] * (1 + tab_v[v]/c)
    """

    wave_decale = wave_template[None, :] * (1.0 + vitesses_decallage[:, None] / (cst.c * 1e-5))

    return wave_decale



def interpoler_template(spectre_template, wave_template, wavelength):
    """
    Interpole le template décalé Doppler pour chaque une grille de vitesses sur la grille de longueur d'onde réelle de chaque ordre

    Entrée :
        spectre_template -> tableau 1D (n_waves_template,), valeurs du template au repos
        wave_template    -> tableau 2D (n_vitesse, n_waves_template) longueurs d'onde décalées Doppler pour chaque vitesse
        wavelength       -> tableau 2D (n_ordre, n_pixels), longueur d'onde réelle de chaque ordre

    Sortie :
        template_3D -> tableau 3D (n_vitesse, n_ordre, n_pixels), les points hors de la plage du template sont nan
    """
    n_vitesse, n_waves_template = wave_template.shape
    n_ordre, n_pixels = wavelength.shape

    spectre_template_3D = np.zeros((n_vitesse, n_ordre, n_pixels))

    for n in range(n_ordre):
        for v in range(n_vitesse):
            spectre_template_3D[v, n, :] = np.interp(wavelength[n], wave_template[v], spectre_template, left=np.nan, right=np.nan)

    return spectre_template_3D



def injecter_signal(spec_interpolated, wave_modelA, tab_phase, berv, wave_template, spectre_template, Kp_inject, Vsys_inject, amplitude):
    """
    Injecte un signal planétaire synthétique connu dans spec_interpolated à Kp/Vsys donnés, avec une amplitude amplifiée pour valider tout le traitement
    spec_interpolated est dans le référentiel tellurique/géocentrique, le signal injecté doit donc être décalé du BERV

    Entrée :
        spec_interpolated      -> (n_ordres, n_temps, n_pixels), données netttoyées
        wave_modelA            -> (n_ordres, n_pixels)
        tab_phase              -> (n_temps,)
        berv                   -> (n_temps,), vitesse radiale barycentrique de la Terre au moment de chaque pose (km/s)
        wave_template          -> (n_pixels_template,), grille au repos du template
        spectre_template       -> (n_pixels_template,), déjà convolué + continuum retiré potentiellement
        Kp_inject, Vsys_inject -> le signal "vrai" injecté (km/s)
        amplitude              -> facteur d'amplification

    Sortie :
        spec_injecte -> même forme, avec le signal ajouté
    """
    n_ordres, n_temps, n_pixels = spec_interpolated.shape
    spec_injecte = spec_interpolated.copy()

    vp = Vsys_inject + Kp_inject * np.sin(2 * np.pi * tab_phase) - berv

    for t_idx in range(n_temps):
        wave_decale = wave_template * (1 + vp[t_idx] / (cst.c * 1e-5))
        for n in range(n_ordres):
            signal = np.interp(wave_modelA[n], wave_decale, spectre_template, left=0.0, right=0.0)   # 0 hors plage : pas de faux signal ajouté
            spec_injecte[n, t_idx, :] += amplitude * signal

    return spec_injecte
