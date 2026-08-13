"""
Variante de pipeline() qui injecte un signal planetaire synthetique connu (Kp/Vsys/amplitude choisis) dans les donnees reelles avant
de refaire tout le traitement (PCA, correlation croisee, carte Kp-Vsys), pour verifier que la pipeline retrouve bien ce signal
injecté a la bonne position, avec la bonne amplitude. Sert a valider que la pipeline fonctionne correctement, independamment
de la presence ou non d'un vrai signal dans les donnees.
Certains réglages sont présent en tête du fichier calibration.py .
"""

import numpy as np
from src import affichage as aff
from src import traitement as t
from src import simulation as s
from src import carte
from src import gestionnaire as g
from src.calibration import calibration



def test_injection(planete, fichier, element, Kp_inject, Vsys_inject, amplitude_inject, configuration=None):
    """
    Entrée :
            plante                 : (dictionnaire) dicionnaire d'information sur la planete concerné (a placer dans simulation.py)
            fichier                :  (string) nom du .pickel (sans l'extension) à exploiter (a placer dans /data)
            element                : (string) mollecule chimique à chercher
            Kp_inject, Vsys_inject : (float) Kp/Vsys absolus (km/s) du signal synthétique injecté
            amplitude_inject       : (float) facteur d'amplification du signal injecté
            configuration          : (optionnel, dictionnaire) configuration à utilisé pour la pipline, si non indiqué, appel de calibration pour touver les paramètres optimaux
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

    g.gestion_dossiers_injection(fichier) # crée les dossier necessaires
    parametres_appel = dict(planete=planete["name"], fichier=fichier, element=element, Kp_inject=Kp_inject, Vsys_inject=Vsys_inject, amplitude_inject=amplitude_inject)
    g.ecrire_configuration(configuration, parametres_appel, f"result/{fichier}/Injection/Configurations/{element}.txt") # imprime sous .txt la configuration utilisé + les parametres d'appel



    # ------------------------------ Traitement des données ------------------------------

    # === Récuperation données ===
    data = g.ouvrir(fichier)

    wave_modelA = data["wave_modelA"]
    spec_interpolated = data["spec_interpolated"]
    err = data["err"]
    bjd_tdb = data["bjd_tdb"] * 86400.0     # [bjd_tdb_secondes]
    berv = data["berv"] * 1e-3              # [m/s] -> [km/s]
    tell = data["tell"]


    phases = t.phases_transit(bjd_tdb, P, Tc)
    u1, u2, epsilon_lineaire = s.coefficients_assombrissement(planete, wave_modelA)
    poids_transit, _ = t.transit(bjd_tdb, planete, u1, u2)


    # === Nettoyage des données brutes ===
    spec_interpolated_normal, diviseur_normal = t.normaliser_tableau_3D(spec_interpolated)
    err_normal, _ = t.normaliser_tableau_3D(err, diviseur_normal)

    spec_interpolated_nettoye, diviseur_mediane = t.nettoyage_tableau_3D(spec_interpolated_normal)
    err_nettoye, _ = t.nettoyage_tableau_3D(err_normal, diviseur_mediane)


    # === Masquer les nan, les bandes telluriques profondes et les raies qui se décalent ===
    """ Paramètres 1/8 et 2/8 et 3/8"""
    masque_tellurique = t.masque_tellurique(tell, pourcentage_bas, pourcentage_haut, largeur_voisinage)
    masque_decallage_spec = t.masque_decallage(spec_interpolated_nettoye, seuil_decallage)
    masque_nan_spec_interpolated = t.masque_nan(spec_interpolated_nettoye)
    masque_nan_err = t.masque_nan(err_nettoye)

    masque = t.fusionner_masques(masque_tellurique, masque_decallage_spec)
    masque = t.fusionner_masques(masque, masque_nan_spec_interpolated)
    masque = t.fusionner_masques(masque, masque_nan_err)

    err_masque = t.appliquer_masque_nan(err_nettoye, masque) # masque_nan car 0 pose probleme pour la division dans la CCF


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

    # === Injecte le signal connu, puis refait la PCA sur les données injectées ===
    spec_injecte = s.injecter_signal(spec_interpolated_nettoye, wave_modelA, phases, berv, wave_template, spectre_template_arrange, Kp_inject, Vsys_inject, amplitude_inject)
    spec_injecte_masque = t.appliquer_masque_nul(spec_injecte, masque)   # reutilise le meme masque
    U_inj, Sigma_inj, V_transp_inj, w_barr_inj = t.pca_3D_decompose(spec_injecte_masque)
    spec_4D_pour_ccf = t.pca_3D_to_4D(U_inj, Sigma_inj, V_transp_inj, w_barr_inj, r_max)

    if propagation_erreurs:
        err_masque = t.propager_erreur((U_inj, Sigma_inj, V_transp_inj, w_barr_inj, masque), err_masque, r_max)


    """ Paramètres 7/8"""
    CCF = carte.correlation_croisee(spec_4D_pour_ccf[:,:,:,-1], err_masque, spectre_template_3D, supprimer_mediane)
    aff.afficher_ccf(CCF, vitesses_decallage, phases, f"{fichier}/Injection/CCF/{element}")

    """ Paramètres 8/8"""
    carte_somme = carte.carte_somme(CCF, Vsys, vitesses_decallage, phases, berv, tab_kp, tab_vsys, poids_transit)
    carte_kp_vsys = carte.somme_to_zscore(carte_somme, tab_vsys, largeur_exclusion)
    aff.afficher_carte_kp_vsys(carte_kp_vsys, tab_kp, tab_vsys, f"{fichier}/Injection/Carte/{element}", Kp)
    aff.afficher_tranche_kp(carte_kp_vsys, tab_kp, tab_vsys, Kp, f"{fichier}/Injection/Tranche_carte/{element}")


    # ------------------------------ Affichage des résultats ------------------------------

    ikp_inj = int(np.argmin(np.abs(tab_kp - Kp_inject)))
    ivs_inj = int(np.argmin(np.abs(tab_vsys - (Vsys_inject - Vsys))))
    ikp_max, ivs_max = np.unravel_index(np.nanargmax(carte_kp_vsys), carte_kp_vsys.shape)
    Vsys_max = Vsys + tab_vsys[ivs_max]
    dKp = tab_kp[ikp_max] - Kp_inject
    dVsys = Vsys_max - Vsys_inject
    ligne = (f"Signal injecté Kp={Kp_inject}, Vsys={Vsys_inject}  ->  "
                f"{carte_kp_vsys[ikp_inj, ivs_inj]:6.2f} sigma au point injecté  |  max de la carte : {carte_kp_vsys[ikp_max, ivs_max]:6.2f} sigma "
                f"à Kp={tab_kp[ikp_max]:.2f}, Vsys={Vsys_max:.2f} (écart dKp={dKp:+.2f}, dVsys={dVsys:+.2f})")
    print(ligne)


    return CCF, vitesses_decallage, phases, berv, Vsys, tab_kp, tab_vsys, largeur_exclusion, Kp, element, poids_transit



test_injection(s.GJ_1214_b, "220311_GJ1214", "CH4", Kp_inject=100, Vsys_inject=15, amplitude_inject=1)
