"""
Fichier principal de la pipeline HRCCS : traitement complet d'un jeu de donnees (nettoyage, masquage, PCA, correlation croisee, 
carte de significativite Kp-Vsys en z-score) pour detecter une molecule dans l'atmosphere d'une exoplanete en transit. 
Contient une configuration par defaut pour éviter la calibration (non recommandé) et des exemples d'appels d'execution en bas de fichier.
Une calibration dure approximativement 2 minutes. Une execussion du main dure, HORS calibration, quelques secondes, pour obtenir 
seulement les cartes CCF, Kp-Vsys et tranche de la Kp-Vsys à Kp fixé Kp théorqiue. Une execussion du main, HORS calibration, dure
1 minute 30, pour obtenir toutes les figures. Ainsi une execussion du main en affichant tout dure au totale 3 minutes 30.
Certains réglages sont présent en tête du fichier calibration.py .
"""

import numpy as np
from src import affichage as aff
from src import traitement as t
from src import simulation as s
from src import carte
from src import gestionnaire as g
from src.fusion_donnee import fusionner_nuits
from src.calibration import calibration


configuration_defaut = dict(
    tab_kp=np.arange(-150, 151, 2),     # en km/s, grille d'ecarts autour du Kp connu
    tab_vsys=np.arange(-100, 101, 1),   # en km/s, grille d'ecarts autour du Vsys connu
    seuil_decallage=3.0,                # masque_decallage : masque les raies qui se décalent entre les prises A et B
    propagation_erreurs=False,          # propager_erreur : propage rigoureusement l'effet de la PCA sur les erreurs recues dans le .pickel
    elargissement_rotationnel=True,     # spectre_pRT_to_CRIRES : applique l'elargissement rotationnel au template (profil de Gray, rotation supposee synchrone)
    abondance=1e-3,                     # spectre_transmission : fraction massique de la molecule recherchee dans l'atmosphere (mass_fraction petitRADTRANS)

    pourcentage_bas=10,                 # masquage tellurique : profondeur minimale masquee
    pourcentage_haut=60,                # masquage tellurique : profondeur maximale masquee (cycle d'hysteresis lorsque pourcentage_bas atteint)
    largeur_voisinage=20,               # masquage tellurique : largeur du voisinage etendu autour des raies profondes pour l'echelle de pourcentage d'intensité
    r_max=8,                            # pca_3D_to_4D : nombre de composantes PCA retirees (bruit/tellurique residuel)
    supprimer_continum=True,            # spectre_pRT_to_CRIRES : retire le continuum du template avant correlation
    facteur_continum=30,                # spectre_pRT_to_CRIRES : largeur de la convolution utilisee pour retirer le continuum
    supprimer_mediane=True,             # correlation_croisee : retire la mediane de la CCF (par vitesse) avant de sommer
    largeur_exclusion=15,               # somme_to_zscore : demi-largeur exclue autour du Vsys testé pour estimer le bruit local
)


def pipeline(planete, fichier, element, tout_afficher = False, configuration = None):
    """
    Entrée :
            plante        : (dictionnaire) dicionnaire d'information sur la planete concerné (a placer dans simulation.py)
            fichier       : (string) nom du .pickel (sans l'extension) à exploiter (a placer dans /data)
            element       : (string) mollecule chimique à chercher
            tout_afficher : (optiannel, bool) permet d'afficher ou pas toutes les figure, False affiche que la carte CCF et la Kp-Vsys pour gagner du temps d'exécution
            configuration : (optionnel, dictionnaire) configuration à utilisé pour la pipline, si non indiqué, appel de calibration pour touver les paramètres optimaux
    Sortie :
            données necessaires pour fusionner plusieurs .pickel, notamment via fusionner_nuits
    """
    # ------------------------------ Parametrage ------------------------------

    Vsys = planete["Vsys"]
    Kp = planete["Kp"]
    Tc = planete["Tc"]
    P = planete["P"]
    v_rot = planete["v_rot"]

    if configuration is None:
        configuration = calibration(planete, fichier, element)

    # === Paramètres fixes ===
    tab_kp = s.grille_kp(Kp, configuration["tab_kp"])  # centre sur Kp + exclut les Kp negatifs
    tab_vsys = configuration["tab_vsys"]
    seuil_decallage = configuration["seuil_decallage"]
    propagation_erreurs = configuration["propagation_erreurs"]
    elargissement_rotationnel = configuration["elargissement_rotationnel"]
    abondance = configuration["abondance"]


    # === Paramètres à calibrer ===
    pourcentage_bas = configuration["pourcentage_bas"]
    pourcentage_haut = configuration["pourcentage_haut"]
    largeur_voisinage = configuration["largeur_voisinage"]
    r_max = configuration["r_max"]
    supprimer_continum = configuration["supprimer_continum"]
    facteur_continum = configuration["facteur_continum"]
    supprimer_mediane = configuration["supprimer_mediane"]
    largeur_exclusion = configuration["largeur_exclusion"]


    g.gestion_dossiers(fichier, element) # crée les dossier necessaires
    parametres_appel = dict(planete=planete["name"], fichier=fichier, element=element, tout_afficher=tout_afficher)
    g.ecrire_configuration(configuration, parametres_appel, f"result/{fichier}/Configurations/{element}.txt") # imprime sous .txt la configuration utilisé + les parametres d'appel



    # ------------------------------ Traitement des données ------------------------------

    # === Récuperation données ===
    data = g.ouvrir(fichier)

    wave_modelA = data["wave_modelA"]
    spec_interpolated = data["spec_interpolated"]
    err = data["err"]
    snr = data["snr"]
    airmass = data["airmass"]
    bjd_tdb = data["bjd_tdb"] * 86400.0     # [bjd_tdb_secondes]
    berv = data["berv"] * 1e-3              # [m/s] -> [km/s]
    tell = data["tell"]


    phases = t.phases_transit(bjd_tdb, P, Tc)
    u1, u2, epsilon_lineaire = s.coefficients_assombrissement(planete, wave_modelA)
    poids_transit, en_transit = t.transit(bjd_tdb, planete, u1, u2)


    if tout_afficher :
        aff.afficher_tableau_3D(spec_interpolated, wave_modelA, phases, f"{fichier}/spec_interpolated_brute", "Spectre temporel")
        aff.afficher_tableau_3D(err, wave_modelA, phases, f"{fichier}/err_brute", "Erreur")
        aff.afficher_snr(snr, phases, f"{fichier}/SNR")
        aff.afficher_airmass(airmass, phases, f"{fichier}/airmass")
        aff.afficher_tell(tell, wave_modelA, f"{fichier}/tell")
        aff.afficher_transit(poids_transit, en_transit, phases, f"{fichier}/transit")

    # === Nettoyage des données brutes ===
    spec_interpolated_normal, diviseur_normal = t.normaliser_tableau_3D(spec_interpolated)
    err_normal, _ = t.normaliser_tableau_3D(err, diviseur_normal)

    if tout_afficher :
        aff.afficher_tableau_3D(spec_interpolated_normal, wave_modelA, phases, f"{fichier}/spec_interpolated_nettoye", "Spectre temporel nettoyé")
        aff.afficher_tableau_3D(err_normal, wave_modelA, phases, f"{fichier}/err_nettoye", "Erreurs nettoyées")

    spec_interpolated_nettoye, diviseur_mediane = t.nettoyage_tableau_3D(spec_interpolated_normal)
    err_nettoye, _ = t.nettoyage_tableau_3D(err_normal, diviseur_mediane)

    if tout_afficher :
        aff.afficher_tableau_3D(spec_interpolated_nettoye, wave_modelA, phases, f"{fichier}/spec_interpolated_nettoye", "Spectre temporel nettoyé")
        aff.afficher_tableau_3D(err_nettoye, wave_modelA, phases, f"{fichier}/err_nettoye", "Erreurs nettoyées")


    # === Masquer les nan, les bandes telluriques profondes et les raies qui se décalent ===
    """ Paramètres 1/8 et 2/8 et 3/8"""
    masque_tellurique = t.masque_tellurique(tell, pourcentage_bas, pourcentage_haut, largeur_voisinage)
    masque_decallage_spec = t.masque_decallage(spec_interpolated_nettoye, seuil_decallage)
    masque_nan_spec_interpolated = t.masque_nan(spec_interpolated_nettoye)
    masque_nan_err = t.masque_nan(err_nettoye)

    masque = t.fusionner_masques(masque_tellurique, masque_decallage_spec)
    masque = t.fusionner_masques(masque, masque_nan_spec_interpolated)
    masque = t.fusionner_masques(masque, masque_nan_err)

    spec_interpolated_masque = t.appliquer_masque_nul(spec_interpolated_nettoye, masque) # masque_nul pour ne pas bloquer la SVD
    err_masque = t.appliquer_masque_nan(err_nettoye, masque) # masque_nan car 0 pose probleme pour la division dans la CCF

    if tout_afficher :
        aff.afficher_tableau_3D(t.appliquer_masque_nan(spec_interpolated_nettoye, masque), wave_modelA, phases, f"{fichier}/spec_interpolated_masqué", "Spectre temporel masqué")
        aff.afficher_tableau_3D(t.appliquer_masque_nan(err_nettoye, masque), wave_modelA, phases, f"{fichier}/err_masqué", "Erreurs masquées") 


    # === PCA ===
    U, Sigma, V_transp, w_barr = t.pca_3D_decompose(spec_interpolated_masque)
    """ Paramètres 4/8"""
    spec_interpolated_4D = t.pca_3D_to_4D(U, Sigma, V_transp, w_barr, r_max) # la 4ème dimension corréspond au r_max utilisé dans la PCA

    if tout_afficher :
        aff.afficher_evolution_pca(spec_interpolated_4D, wave_modelA, phases, f"{fichier}", element)
        aff.afficher_tableau_3D(t.appliquer_masque_nan(spec_interpolated_4D[:,:,:,-1], masque), wave_modelA, phases, f"{fichier}/spectre_temporelle_pca") # Affiche tous les spectres temporelles pour le r_max

    if propagation_erreurs:
        err_masque = t.propager_erreur((U, Sigma, V_transp, w_barr, masque), err_masque, r_max)



    # ------------------------------ Simulation des données ------------------------------

    vitesses_decallage = s.vitesse_decallage(phases, Vsys, tab_vsys, tab_kp, berv)
    lambda_min, lambda_max = s.bornes_wavelength(wave_modelA, vitesses_decallage)

    atmosphere = s.creer_atmosphere([lambda_min, lambda_max], element)
    wave_template, spectre_template = s.spectre_transmission(atmosphere, planete, element, abondance)

    """ Paramètres 5/8 et 6/8"""
    spectre_template_arrange = s.spectre_pRT_to_CRIRES(wave_template, spectre_template, supprimer_continum, facteur_continum, elargissement_rotationnel, v_rot, epsilon_lineaire)
    wave_template_decalle_2D = s.doppler(wave_template, vitesses_decallage)
    spectre_template_3D = s.interpoler_template(spectre_template_arrange, wave_template_decalle_2D, wave_modelA)



    # ------------------------------ Match des données ------------------------------

    """ Paramètres 7/8"""
    CCF = carte.correlation_croisee(spec_interpolated_4D[:,:,:,-1], err_masque, spectre_template_3D, supprimer_mediane)
    aff.afficher_ccf(CCF, vitesses_decallage, phases, f"{fichier}/CCF/{element}")

    
    carte_somme = carte.carte_somme(CCF, Vsys, vitesses_decallage, phases, berv, tab_kp, tab_vsys, poids_transit)
    """ Paramètres 8/8"""
    carte_zscore = carte.somme_to_zscore(carte_somme, tab_vsys, largeur_exclusion)
    aff.afficher_carte_kp_vsys(carte_zscore, tab_kp, tab_vsys, f"{fichier}/Carte/{element}", Kp)
    aff.afficher_tranche_kp(carte_zscore, tab_kp, tab_vsys, Kp, f"{fichier}/Tranche_carte/{element}")



    return CCF, vitesses_decallage, phases, berv, Vsys, tab_kp, tab_vsys, largeur_exclusion, Kp, element, poids_transit



# ------------------------------ Exécutions ------------------------------

nuits =[]
nuits.append(pipeline(s.GJ_1214_b, "220311_GJ1214", "H2O"))
nuits.append(pipeline(s.GJ_1214_b, "220330_GJ1214", "H2O"))
nuits.append(pipeline(s.GJ_1214_b, "220526_GJ1214", "H2O"))
nuits.append(pipeline(s.GJ_1214_b, "220703_GJ1214", "H2O"))
nuits.append(pipeline(s.GJ_1214_b, "220810_GJ1214", "H2O"))
nuits.append(pipeline(s.GJ_1214_b, "230622_GJ1214", "H2O"))
nuits.append(pipeline(s.GJ_1214_b, "230810_GJ1214", "H2O"))
nuits.append(pipeline(s.GJ_1214_b, "240806_GJ1214", "H2O"))
nuits.append(pipeline(s.GJ_1214_b, "240817_GJ1214", "H2O"))

fusionner_nuits(nuits, "GJ1214_b_fusionne")


mollecules = ["C2H2", "CH4", "CO", "CO2","HCN" ,"NH3", "HDO"]


for element in mollecules :
    nuits =[]
    nuits.append(pipeline(s.GJ_1214_b, "220311_GJ1214", element))
    nuits.append(pipeline(s.GJ_1214_b, "220330_GJ1214", element))
    nuits.append(pipeline(s.GJ_1214_b, "220526_GJ1214", element))
    nuits.append(pipeline(s.GJ_1214_b, "220703_GJ1214", element))
    nuits.append(pipeline(s.GJ_1214_b, "220810_GJ1214", element))
    nuits.append(pipeline(s.GJ_1214_b, "230622_GJ1214", element))
    nuits.append(pipeline(s.GJ_1214_b, "230810_GJ1214", element))
    nuits.append(pipeline(s.GJ_1214_b, "240806_GJ1214", element))
    nuits.append(pipeline(s.GJ_1214_b, "240817_GJ1214", element))

    fusionner_nuits(nuits, "GJ1214_b_fusionne")
