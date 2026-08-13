"""
Détermine la limite supérieure sur l'abondance (fraction massique) d'une molécule non détectée dans les données réelles. Pour cela, on 
injecte un signal synthétique à la position (Kp, Vsys) réelle de la planète avec une abondance candidate, on refait tout le traitement 
(PCA, corrélation croisée, carte Kp-Vsys), et on lit le z-score obtenu au point injecté. Cette recherche par dichotomie en échelle log 
sur l'abondance trouve ainsi la plus petite abondance qui aurait été détectée à seuil_sigma.
Certains réglages sont présent en tête du fichier calibration.py .
"""

import numpy as np
from src import affichage as aff
from src import traitement as t
from src import simulation as s
from src import carte
from src import gestionnaire as g
from src.calibration import calibration



def limite_abondance(planete, fichier, element, seuil_sigma=5.0, abondance_min=1e-8, abondance_max=1e-1, tolerance_log=0.05, max_iter=30, configuration=None):
    """
    Entrée :
            planete       : (dictionnaire) dicionnaire d'information sur la planete concerné (a placer dans simulation.py)
            fichier       : (string) nom du .pickel (sans l'extension) à exploiter (a placer dans /data)
            element       : (string) mollecule chimique dont on cherche la limite d'abondance
            seuil_sigma   : (float) seuil de détection (en sigma) au delà duquel le signal injecté est considéré "détecté"
            abondance_min, abondance_max : (float) bornes de la recherche par dichotomie (fraction massique)
            tolerance_log : (float) précision visée sur log10(abondance) pour arrêter la dichotomie
            max_iter      : (int) nombre maximal d'itérations de la dichotomie (sécurité)
            configuration : (optionnel, dictionnaire) configuration à utiliser pour la pipeline, si non indiqué, appel de calibration
    Sortie :
            abondance_limite : (float) limite supérieure trouvée sur l'abondance (fraction massique)
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

    # === Paramètres à calibrer ===
    pourcentage_bas = configuration["pourcentage_bas"]
    pourcentage_haut = configuration["pourcentage_haut"]
    largeur_voisinage = configuration["largeur_voisinage"]
    r_max = configuration["r_max"]
    supprimer_continum = configuration["supprimer_continum"]
    facteur_continum = configuration["facteur_continum"]
    supprimer_mediane = configuration["supprimer_mediane"]
    largeur_exclusion = configuration["largeur_exclusion"]

    g.gestion_dossiers_limite(fichier)  # crée les dossiers necessaires
    parametres_appel = dict(planete=planete["name"], fichier=fichier, element=element, seuil_sigma=seuil_sigma, abondance_min=abondance_min, abondance_max=abondance_max, tolerance_log=tolerance_log)
    g.ecrire_configuration(configuration, parametres_appel, f"result/{fichier}/Limite_abondance/Configurations/{element}.txt")



    # ------------------------------ Traitement des données (indépendant de l'abondance) ------------------------------

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

    spec_interpolated_normal, diviseur_normal = t.normaliser_tableau_3D(spec_interpolated)
    err_normal, _ = t.normaliser_tableau_3D(err, diviseur_normal)

    spec_interpolated_nettoye, diviseur_mediane = t.nettoyage_tableau_3D(spec_interpolated_normal)
    err_nettoye, _ = t.nettoyage_tableau_3D(err_normal, diviseur_mediane)

    masque_tellurique = t.masque_tellurique(tell, pourcentage_bas, pourcentage_haut, largeur_voisinage)
    masque_decallage_spec = t.masque_decallage(spec_interpolated_nettoye, seuil_decallage)
    masque_nan_spec_interpolated = t.masque_nan(spec_interpolated_nettoye)
    masque_nan_err = t.masque_nan(err_nettoye)

    masque = t.fusionner_masques(masque_tellurique, masque_decallage_spec)
    masque = t.fusionner_masques(masque, masque_nan_spec_interpolated)
    masque = t.fusionner_masques(masque, masque_nan_err)

    err_masque_brut = t.appliquer_masque_nan(err_nettoye, masque)  # masque_nan car 0 pose probleme pour la division dans la CCF

    vitesses_decallage = s.vitesse_decallage(phases, Vsys, tab_vsys, tab_kp, berv)
    lambda_min, lambda_max = s.bornes_wavelength(wave_modelA, vitesses_decallage)
    atmosphere = s.creer_atmosphere([lambda_min, lambda_max], element)  # ne depend pas de l'abondance, cree une seule fois

    ikp_inj = int(np.argmin(np.abs(tab_kp - Kp)))
    ivs_inj = int(np.argmin(np.abs(tab_vsys - 0.0)))  # Comme on à centrée en 0 le vrai Vsys



    # ------------------------------ Evaluation du z-score pour une abondance donnée ------------------------------

    historique_abondances, historique_zscores = [], []  # garde tous les points testes pour tracer z-score(abondance)

    def zscore_pour_abondance(abondance):
        wave_template, spectre_template = s.spectre_transmission(atmosphere, planete, element, abondance)
        spectre_template_arrange = s.spectre_pRT_to_CRIRES(wave_template, spectre_template, supprimer_continum, facteur_continum, elargissement_rotationnel, v_rot, epsilon_lineaire)
        wave_template_decalle_2D = s.doppler(wave_template, vitesses_decallage)
        spectre_template_3D = s.interpoler_template(spectre_template_arrange, wave_template_decalle_2D, wave_modelA)

        spec_injecte = s.injecter_signal(spec_interpolated_nettoye, wave_modelA, phases, berv, wave_template, spectre_template_arrange, Kp, Vsys, 1.0)  # amplitude = 1 : l'abondance seule controle l'intensite du signal
        spec_injecte_masque = t.appliquer_masque_nul(spec_injecte, masque)
        U_inj, Sigma_inj, V_transp_inj, w_barr_inj = t.pca_3D_decompose(spec_injecte_masque)
        spec_4D_pour_ccf = t.pca_3D_to_4D(U_inj, Sigma_inj, V_transp_inj, w_barr_inj, r_max)

        err_masque = err_masque_brut
        if propagation_erreurs:
            err_masque = t.propager_erreur((U_inj, Sigma_inj, V_transp_inj, w_barr_inj, masque), err_masque, r_max)

        CCF = carte.correlation_croisee(spec_4D_pour_ccf[:, :, :, -1], err_masque, spectre_template_3D, supprimer_mediane)
        carte_somme = carte.carte_somme(CCF, Vsys, vitesses_decallage, phases, berv, tab_kp, tab_vsys, poids_transit)
        carte_kp_vsys = carte.somme_to_zscore(carte_somme, tab_vsys, largeur_exclusion)

        z = carte_kp_vsys[ikp_inj, ivs_inj]
        historique_abondances.append(abondance)
        historique_zscores.append(z)
        return z, carte_kp_vsys



    # ------------------------------ Dichotomie en échelle log ------------------------------

    print(f"\n--- limite superieure sur l'abondance de {element} (seuil = {seuil_sigma} sigma) ---")

    log_min, log_max = np.log10(abondance_min), np.log10(abondance_max)

    z_min, _ = zscore_pour_abondance(abondance_min)
    print(f"  abondance={abondance_min:.3e} | z={z_min:6.2f} sigma  (borne basse)")
    if z_min >= seuil_sigma:
        print(f"  => deja detecte a la borne basse : diminue abondance_min pour encadrer la vraie limite")
        aff.afficher_zscore_abondance(historique_abondances, historique_zscores, seuil_sigma, abondance_min, f"{fichier}/Limite_abondance/Zscore/{element}")               
        return abondance_min

    z_max, carte_finale = zscore_pour_abondance(abondance_max)
    print(f"  abondance={abondance_max:.3e} | z={z_max:6.2f} sigma  (borne haute)")
    if z_max < seuil_sigma:
        print(f"  => jamais detecte meme a la borne haute : augmente abondance_max pour encadrer la vraie limite")
        aff.afficher_zscore_abondance(historique_abondances, historique_zscores, seuil_sigma, abondance_max, f"{fichier}/Limite_abondance/Zscore/{element}")
        return abondance_max

    for i in range(max_iter):
        if log_max - log_min < tolerance_log:
            break
        log_mid = 0.5 * (log_min + log_max)
        abondance_mid = 10 ** log_mid
        z_mid, carte_mid = zscore_pour_abondance(abondance_mid)
        detecte = z_mid >= seuil_sigma
        etat = ">= seuil" if detecte else "<  seuil"
        print(f"  iteration {i + 1:2d} | abondance={abondance_mid:.3e} | z={z_mid:6.2f} sigma  {etat}")
        if detecte:
            log_max, carte_finale = log_mid, carte_mid
        else:
            log_min = log_mid

    abondance_limite = 10 ** log_max
    print(f"  => abondance limite ({seuil_sigma} sigma) : {abondance_limite:.3e}")



    # ------------------------------ Sauvegarde des résultats ------------------------------

    aff.afficher_carte_kp_vsys(carte_finale, tab_kp, tab_vsys, f"{fichier}/Limite_abondance/Carte/{element}", Kp)
    aff.afficher_tranche_kp(carte_finale, tab_kp, tab_vsys, Kp, f"{fichier}/Limite_abondance/Tranche_carte/{element}")
    aff.afficher_zscore_abondance(historique_abondances, historique_zscores, seuil_sigma, abondance_limite, f"{fichier}/Limite_abondance/Zscore/{element}")


    return abondance_limite



mollecules = ["C2H2", "CH4", "CO", "CO2", "H2O","HCN" ,"NH3", "HDO"]

for element in mollecules :
    limite_abondance(s.GJ_1214_b, "220311_GJ1214", element)
    limite_abondance(s.GJ_1214_b, "220330_GJ1214", element)
    limite_abondance(s.GJ_1214_b, "220526_GJ1214", element)
    limite_abondance(s.GJ_1214_b, "220703_GJ1214", element)
    limite_abondance(s.GJ_1214_b, "220810_GJ1214", element)
    limite_abondance(s.GJ_1214_b, "230622_GJ1214", element)
    limite_abondance(s.GJ_1214_b, "230810_GJ1214", element)
    limite_abondance(s.GJ_1214_b, "240806_GJ1214", element)
    limite_abondance(s.GJ_1214_b, "240817_GJ1214", element)
