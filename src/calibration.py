"""
Calibre automatiquement les parametres de la pipeline (masquage tellurique, PCA, largeur d'exclusion, etc.) pour un triplet
(planete, fichier, element) donné, en testant une grille de valeurs et en notant chacune via un signal synthetique injecte a
plusieurs positions Kp-Vsys. Renvoie un dictionnaire "configuration" pret a etre utilise par pipeline()/test_injection().
Certains paramètres de la pipeline doivent être ajusté ici, ils sont dans "Parametres de pipline"
Une calibration dure approximativement 2 minutes

Pour un jeu de paramètres et une position d'injection (Kp_i, Vsys_i) donnée, un score_i est attribué par multiplication de
3 critères : 
    1) Correlation gausienne A (tache localisée proche du point Kp-Vsys réel, étendue spatialement raisonnable)
    2) Rétention (information pas trop perdue à cause de la PCA)
    3) Localisation d (proximité entre le maximum global de la carte et le Kp réel)
Le score final est la moyenne des score_i sur toutes les positions testées. 

Attention, score = moyenne(score_i) et non moyenne(A) x moyenne(retention) x moyenne(d), les deux quantités sont différentes.
r_max est calibré une première fois au début (car c'est le paramètre le plus influent), puis une seconde fois à la fin
une fois les autres paramètres fixés.
"""

import warnings
import numpy as np
from src import traitement as t
from src import simulation as s
from src import carte
from src import gestionnaire as g


# -------------- Parametres de pipline --------------

# Grilles d'essai (Kp,Vsys) utilisées pendant la calibration. Valeurs autour des Kp-Vsys réel connu inscrit dans le dictionnaire dans simulation.py, en km/s
TAB_KP = np.arange(-150, 151, 2)
TAB_VSYS = np.arange(-60, 61, 2)


# Seuil de masque_decallage : un pixel est masqué si son écart-type temporel dépasse ce facteur fois l'écart-type médian de son ordre (gère les spectre qui se decalle entre observations A et B)
SEUIL_DECALLAGE = 3.0

# propager_erreur : propage rigoureusement l'effet de la PCA sur les erreurs au lieu de garder l'erreur brute masquee (expérimental)
PROPAGATION_ERREURS = False

# spectre_pRT_to_CRIRES : applique l'elargissement rotationnel au template (profil de Gray, rotation supposee synchrone)
ELARGISSEMENT_ROTATIONNEL = True

# spectre_transmission : fraction massique (mass_fraction) de la molecule recherchee dans l'atmosphere, utilisee par petitRADTRANS pour generer le template
ABONDANCE = 1e-3


# -------------- Parametres de Calibration --------------

# Écarts (dKp, dVsys) en km/s par rapport au (Kp, Vsys) réel de la planète, positions de test pour la calibration.
# Plusieurs positions car injecter sur une bosse de bruit peut augmenter un optimum qui n'existe pas.
POSITIONS = [(-7.0, 25.0), (-50, -30.0), (60, 50.0), (20, -58.0)]


# Amplitudes d'injection (multiplicateur de l'amplitude physique normale du template, 1.0 = amplitude normale), testées dans l'ordre croissant
# En début de calibration : on garde la plus petite qui donne un score de départ >0.1 sur la configuration de départ
AMPLITUDES_CALIBRATION = [1.0, 2.0, 5.0, 10.0, 20.0]

# Configuration initiale
DEPART = dict(pourcentage_bas=10, pourcentage_haut=60, largeur_voisinage=20, r_max=8,
              supprimer_continum=True, facteur_continum=30, supprimer_mediane=True,
              largeur_exclusion=15)


# Valeurs testées pour chaque paramètre
GRILLE_ESSAIS = dict(
    r_max=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14],   # pca_3D_to_4D : nombre de composantes PCA retirees (bruit/tellurique residuel)
    supprimer_continum=[True, False],             # spectre_pRT_to_CRIRES : retire le continuum du template avant correlation
    facteur_continum=[10, 20, 30, 50, 80, 120],   # spectre_pRT_to_CRIRES : largeur de la convolution utilisee pour retirer le continuum
    pourcentage_bas=[0, 5, 10, 15, 25, 35],       # masquage tellurique : profondeur minimale masquee
    pourcentage_haut=[20, 40, 60, 80],            # masquage tellurique : profondeur maximale masquee (cycle d'hysteresis lorsque pourcentage_bas atteint)
    largeur_voisinage=[5, 10, 20, 35, 50],        # masquage tellurique : largeur du voisinage etendu autour des raies profondes pour l'echelle de pourcentage d'intensité
    supprimer_mediane=[True, False],              # correlation_croisee : retire la mediane de la CCF (par vitesse) avant de sommer
    largeur_exclusion=[3, 5, 8, 10, 15],          # somme_to_zscore : demi-largeur exclue autour du Vsys testé pour estimer le bruit local
)


# Résolution du spectromètre
RESOLUTION = 100000


# Puissance appliquee a exp(-d^2/2) dans le score : si supérieur à 1, accentue le poids dans le score finale d'une faible distance entre le max global et à Kp réel fixé
PUISSANCE_LOCALITE = 2



def calibration(planete, fichier, element):
    """
    Entrée :
        planete -> dictionnaire planète (défini dans src/simulation.py)
        fichier -> nom du pickle (dans data/), sans l'extension
        element -> molécule à calibrer (dépend des tables d'opacités petitRADTRANS installées)

    Sortie :
        configuration -> dict {tab_kp, tab_vsys, seuil_decallage, pourcentage_bas,
                                pourcentage_haut, largeur_voisinage, r_max, supprimer_continum,
                                facteur_continum, supprimer_mediane, largeur_exclusion,
                                propagation_erreurs, elargissement_rotationnel, abondance}
    """
    # -------------- Charger données --------------

    Vsys = planete["Vsys"]
    Kp = planete["Kp"]
    Tc = planete["Tc"]
    P = planete["P"]
    v_rot = planete["v_rot"]


    tab_kp = s.grille_kp(Kp, TAB_KP) # grille de Kp avec le vrai Kp en km/s (la grille s'étend de TAB_KP autour de Kp en tronquant les valeurs négatives)
    tab_vsys = TAB_VSYS              # grille d'ecarts autour de Vsys, Vsys est par la suite centré en 0

    # POSITIONS est en écarts (dKp, dVsys) par rapport au (Kp, Vsys) réel : conversion en positions absolues (kp_i, vs_i)
    positions_absolues = [(Kp + dkp, Vsys + dvs) for dkp, dvs in POSITIONS]
    for (dkp, dvs), (kp_i, vs_i) in zip(POSITIONS, positions_absolues):
        assert tab_vsys.min() <= dvs <= tab_vsys.max(), (
            f"POSITIONS contient un dVsys={dvs} hors de TAB_VSYS=[{tab_vsys.min()},{tab_vsys.max()}] "
            f"(position (dKp={dkp}, dVsys={dvs}) -> absolu (Kp={kp_i:.2f}, Vsys={vs_i:.2f})) : "
            f"le signal injecté serait invisible dans la grille testée, élargir TAB_VSYS ou ajuster POSITIONS."
        )

    def charger(): # passer par des définitions permet de ne pas s'embrouiller avec des variables d'étapes partielles
        data = g.ouvrir(fichier)
        bjd_tdb = data["bjd_tdb"] * 86400.0
        phases = t.phases_transit(bjd_tdb, P, Tc)
        spec_n, div = t.normaliser_tableau_3D(data["spec_interpolated"])
        err_n, _ = t.normaliser_tableau_3D(data["err"], div)
        spec, div_med = t.nettoyage_tableau_3D(spec_n)
        err, _ = t.nettoyage_tableau_3D(err_n, div_med)
        berv = data["berv"] * 1e-3  # [m/s] -> [km/s]
        return data["wave_modelA"], spec, err, data["tell"], phases, berv, bjd_tdb
    print("\n" + "=" * 120)
    print(f"Chargement de {fichier} :")
    WAVE, SPEC, ERR, TELL, PHASES, BERV, BJD_TDB = charger()
    N_TEMPS = len(PHASES)
    N_PIXELS = SPEC.shape[2]
    VITESSES = s.vitesse_decallage(PHASES, Vsys, tab_vsys, tab_kp, BERV)
    LAMBDA_MIN, LAMBDA_MAX = s.bornes_wavelength(WAVE, VITESSES)



    # -------------- Informations sur le jeu données --------------

    couverture = (PHASES.max() - PHASES.min()) * P / 3600 # en heures
    sinphi = np.sin(2 * np.pi * PHASES)
    print(f"  {WAVE.shape[0]} ordres, {N_TEMPS} poses, couverture de phase de {couverture:.2f} h, {N_PIXELS} pixels par ordre")
    print(f"  Kp de l'orbite : {Kp:.2f} km/s")
    # Une erreur dKp déplace la trace de dKp * sin(2*pi*phase) ie dKp detectable => dKp > resolution / sin(2*pi*phase)
    resolution = 300000 / RESOLUTION   # élément de résolution CRIRES+ = c/R, en km/s
    dkp_resolu = resolution / np.sqrt(np.mean(sinphi ** 2)) # moyenne quadratique de sinphi
    print(f"  Élément de résolution en Vsys : ~{resolution:.2f} km/s")
    print(f"  Précision maximale atteignable sur Kp : ~{dkp_resolu:.0f} km/s "
          f"(en-dessous, la trace bouge de moins d'un élément de résolution)")
    print("=" * 120 + "\n")

    # Pour comparer les taches de detection des Kp-Vsys à des gausiennes, on se fixe les largeurs :
    sigma_k = dkp_resolu / 2
    sigma_vsys = resolution / 2



    # -------------- Géneration du template de spectre --------------

    SPECTRE_PRT = s.spectre_transmission(s.creer_atmosphere([LAMBDA_MIN, LAMBDA_MAX], element), planete, element, ABONDANCE)
    U1, U2, EPSILON_LINEAIRE = s.coefficients_assombrissement(planete, WAVE)
    POIDS_TRANSIT, _ = t.transit(BJD_TDB, planete, U1, U2)
    AMPLITUDE = AMPLITUDES_CALIBRATION[0]



    # -------------- Gestionnaire des caches (évite de recalculer plusieurs fois un même objet) --------------
    """ Cette partie serait optionnel en soit mais rallongerai de trop la vitesse d'execussion de ce programme de calibration des paramètres.
        Actuellement un appel calibration pour (planete, fichier, element) donnée dure approximativement 2 minutes """

    _cache_masque, _cache_template, _cache_svd, _cache_avant = {}, {}, {}, {}
    MAX_CACHE = 10   # nombre maximum d'objet stocké dans un dictionnaire de cache

    def _ranger(cache, cle, fabriquer):
        if cle in cache:
            cache[cle] = cache.pop(cle)          # mettre a jour le cache en re-plaçant l'element en debut de liste
        else:
            if len(cache) >= MAX_CACHE:
                cache.pop(next(iter(cache)))     # supprimer l'entrée la plus ancienne, next(iter(cache)) = element le plus ancien
            cache[cle] = fabriquer()
        return cache[cle]

    def get_masque(pb, ph, lv, sd): # pourcentages bas, haut, largeur_voisinage et seuil_decallage
        def fab():
            m = t.fusionner_masques(t.masque_tellurique(TELL, pb, ph, lv), t.masque_decallage(SPEC, sd))
            m = t.fusionner_masques(m, t.masque_nan(SPEC))
            return t.fusionner_masques(m, t.masque_nan(ERR))
        return _ranger(_cache_masque, (pb, ph, lv, sd), fab)

    def get_template(sc, fc): # supprmer_continum, facteur_continum
        def fab():
            w, sp = SPECTRE_PRT # waves, spectre
            arrange = s.spectre_pRT_to_CRIRES(w, sp, sc, fc, ELARGISSEMENT_ROTATIONNEL, v_rot, EPSILON_LINEAIRE)
            return s.interpoler_template(arrange, s.doppler(w, VITESSES), WAVE), arrange, w
        return _ranger(_cache_template, (sc, fc), fab)

    def get_svd(cle_masque, sc=None, fc=None, kp_i=None, vs_i=None):    # SVD des données réelles (kp_i=None) ou des données avec signal injecté
        def fab():
            masque = get_masque(*cle_masque)
            if kp_i is None:
                entree = SPEC
            else:
                _, arrange, w = get_template(sc, fc)
                entree = s.injecter_signal(SPEC, WAVE, PHASES, BERV, w, arrange, kp_i, vs_i, AMPLITUDE)
            return t.pca_3D_decompose(t.appliquer_masque_nul(entree, masque))
        return _ranger(_cache_svd, (cle_masque, sc, fc, kp_i, vs_i, AMPLITUDE), fab)

    def get_amplitude_avant(cle_masque, sc, fc, kp_i, vs_i, supprimer_mediane): # Amplitude du signal injecte seul (sans les donnees reelles, sans aucune reduction PCA) au point injecte : mesure ce qui serait recupere si la PCA n'enlevait rien
        def fab():
            masque = get_masque(*cle_masque)
            template_3D, arrange, w = get_template(sc, fc)
            signal_brut = s.injecter_signal(np.zeros_like(SPEC), WAVE, PHASES, BERV, w, arrange, kp_i, vs_i, AMPLITUDE)
            signal_masque = t.appliquer_masque_nul(signal_brut, masque)
            err_masque = t.appliquer_masque_nan(ERR, masque)
            CCF_avant = carte.correlation_croisee(signal_masque, err_masque, template_3D, supprimer_mediane)
            v0 = vs_i - Vsys
            return carte.carte_somme(CCF_avant, Vsys, VITESSES, PHASES, BERV, np.array([kp_i]), np.array([v0]), POIDS_TRANSIT)[0, 0]
        return _ranger(_cache_avant, (cle_masque, sc, fc, kp_i, vs_i, supprimer_mediane, AMPLITUDE), fab)



    # -------------- Évaluation d'un jeu de paramètres --------------

    def evaluer(par): # Renvoi le score moyen obtenue sur 4 positions d'injection testé pour un jeu de paramètre donnée
        cle_masque = (par["pourcentage_bas"], par["pourcentage_haut"], par["largeur_voisinage"], par["seuil_decallage"])
        err_masque = t.appliquer_masque_nan(ERR, get_masque(*cle_masque))
        sc, fc, r = par["supprimer_continum"], par["facteur_continum"], par["r_max"]

        template, _, _ = get_template(sc, fc)

        # residu des donnees reelles (sans injection)
        U_sans_inj, S_sans_inj, V_sans_inj, wb_sans_inj = get_svd(cle_masque)
        residu_sans_inj = t.pca_3D_reconstruire(U_sans_inj, S_sans_inj, V_sans_inj, wb_sans_inj, r)

        # On défini ici notre systeme de notation pour un peu de parametre à une position d'injection donnée, qui est la multiplication de 3 scores différents
        notes, As, retentions, distances = [], [], [], []
        for (kp_i, vs_i) in positions_absolues:
            v0 = vs_i - Vsys  # position de l'injection dans le referentiel "Vsys centrée en 0"

            U, S, V, wb = get_svd(cle_masque, sc, fc, kp_i, vs_i)
            residu_inj = t.pca_3D_reconstruire(U, S, V, wb, r)
            if PROPAGATION_ERREURS:
                err_pour_carte = t.propager_erreur((U, S, V, wb, get_masque(*cle_masque)), err_masque, r)
            else:
                err_pour_carte = err_masque
            CCF_inj = carte.correlation_croisee(residu_inj, err_pour_carte, template, par["supprimer_mediane"])
            carte_somme = carte.carte_somme(CCF_inj, Vsys, VITESSES, PHASES, BERV, tab_kp, tab_vsys, POIDS_TRANSIT)
            carte_kp_vsys = carte.somme_to_zscore(carte_somme, tab_vsys, par["largeur_exclusion"])

            # --- Score 1 (A) : Corrélation entre une gausienne 2D centrée en (Kp_injecté, Vsys_injecté), de largeur (sigma_kp, sigma_vsys) et la carte Kp-Vsys avec signal injecté ---
            gauss_k = np.exp(-(tab_kp - kp_i) ** 2 / (2 * sigma_k ** 2))
            gauss_v = np.exp(-(tab_vsys - v0) ** 2 / (2 * sigma_vsys ** 2))
            gabarit = gauss_k[:, None] * gauss_v[None, :]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                A = np.nansum(carte_kp_vsys * gabarit) / np.sum(gabarit ** 2)

            # --- Score 2 (retention) : amplitude du signal injecte : apres PCA /  avant PCA ---
            CCF_signal = carte.correlation_croisee(residu_inj - residu_sans_inj, err_masque, template, par["supprimer_mediane"])
            amplitude_apres = carte.carte_somme(CCF_signal, Vsys, VITESSES, PHASES, BERV, np.array([kp_i]), np.array([v0]), POIDS_TRANSIT)[0, 0]
            amplitude_avant = get_amplitude_avant(cle_masque, sc, fc, kp_i, vs_i, par["supprimer_mediane"])
            retention = amplitude_apres / amplitude_avant

            # --- Score 3 (localisation) : ecart entre le vrai max de la carte et le point injecte, normalisé par l'élement de résolution de chaque axe ---
            if np.all(np.isnan(carte_kp_vsys)):
                notes.append(0.0); As.append(A); retentions.append(retention); distances.append(np.nan)
                continue
            mk, mv = np.unravel_index(np.nanargmax(carte_kp_vsys), carte_kp_vsys.shape)
            d2 = ((tab_kp[mk] - kp_i) / dkp_resolu) ** 2 + ((tab_vsys[mv] - v0) / (2 * resolution)) ** 2    # distance quadatique normalisé au carré

            score_i = A * retention * (np.exp(-d2 / 2) ** PUISSANCE_LOCALITE)
            notes.append(score_i); As.append(A); retentions.append(retention); distances.append(np.sqrt(d2))

        return float(np.mean(notes)), float(np.mean(As)), float(np.mean(retentions)), float(np.nanmean(distances))



    # -------------- Choix de l'amplitude d'injection --------------

    courant = dict(DEPART)
    courant["seuil_decallage"] = SEUIL_DECALLAGE

    def choisir_amplitude():   # Teste les amplitudes candidates dans l'ordre croissant sur la configuration de depart, garde la premiere qui donne un score stable (> 0)
        nonlocal AMPLITUDE
        print(f"\n--- amplitude d'injection ---")
        for a in AMPLITUDES_CALIBRATION:
            AMPLITUDE = a
            score, A_moy, retention_moy, d_moy = evaluer(courant)
            if score > 5:   # Score minimum arbitraire, pour évité que les score soient trop resséré sur 0 et faussé la calibration
                print(f"  amplitude={a:6.2f}x | score={score:6.3f} | A_moy={A_moy:6.2f} | retention={retention_moy:5.2f} | d_moy={d_moy:5.2f}  <-- retenue")
                print(f"  => amplitude = {a}")
                return
            print(f"  amplitude={a:6.2f}x | score={score:6.3f} | A_moy={A_moy:6.2f} | retention={retention_moy:5.2f} | d_moy={d_moy:5.2f}")
        AMPLITUDE = AMPLITUDES_CALIBRATION[-1]
        print(f"  => aucune amplitude testee ne stabilise le score, on garde la plus grande testee : {AMPLITUDE}")

    choisir_amplitude()



    # -------------- Test tous les jeux de donnés --------------

    def optimiser(nom, valeurs, courant):   # Teste toutes les valeurs d'un paramètre, renvoie le meilleure au sens du score (A x retention x exp(-d²/2)^PUISSANCE_LOCALITE)."""
        print(f"\n--- {nom} ---")
        resultats = []
        for v in valeurs:
            essai = dict(courant)
            essai[nom] = v
            score, A_moy, retention_moy, d_moy = evaluer(essai)
            resultats.append((v, score, A_moy, retention_moy, d_moy))

        scores_valides = [score for _, score, *_ in resultats if not np.isnan(score) and score > 0]
        if scores_valides:
            meilleur_score = max(scores_valides)
            valeur_gagnante = next(v for v, score, *_ in resultats if score == meilleur_score)
        else:
            meilleur_score, valeur_gagnante = 0, DEPART[nom]

        for v, score, A_moy, retention_moy, d_moy in resultats:
            marque = "  <-- meilleur" if (v == valeur_gagnante and meilleur_score > 0) else ""
            print(f"  {nom}={str(v):>6s} | score={score:6.3f} | A_moy={A_moy:6.2f} "
                  f"| retention={retention_moy:5.2f} | d_moy={d_moy:5.2f}{marque}", flush=True)
        print(f"  => {nom} = {valeur_gagnante}")
        return valeur_gagnante



    # -------------- Affichage résultats --------------

    print("\n" + "=" * 190)
    print(f"CALIBRATION -- {fichier} | molécule {element} | {len(POSITIONS)} positions | amplitude {AMPLITUDE}")
    print("critère : score = A (correlation gausienne centrée sur l'injection) x retention (signal recupéré après PCA) x exp(-d²/2)^PUISSANCE_LOCALITE (distance max globale - max à Kp injecté)")
    print("=" * 190)

    score_depart, A_depart, retention_depart, d_depart = evaluer(courant)
    print(f"\nPoint de départ : score = {score_depart:.3f} (A={A_depart:.2f}, retention={retention_depart:.2f}, d={d_depart:.2f})")

    # r_max en premier car c'est le facteur de plus influent
    for nom in ["r_max", "supprimer_continum", "facteur_continum", "pourcentage_bas", "pourcentage_haut",
                "largeur_voisinage", "supprimer_mediane", "largeur_exclusion"]:
        courant[nom] = optimiser(nom, GRILLE_ESSAIS[nom], courant)

    # Second passage : r_max a été choisi avant que les autres paramètres soient réglés, on fait donc un nouveau balayage de r_max avec les autres parametres fixés
    print("\n--- Second passage pour r_max avec les autres paramètres fixés ---")
    courant["r_max"] = optimiser("r_max", GRILLE_ESSAIS["r_max"], courant)

    score_final, A_final, retention_final, d_final = evaluer(courant)

    print("\n" + "=" * 100)
    print("PARAMÈTRES RECOMMANDÉS")
    print("=" * 100)
    for nom in ["pourcentage_bas", "pourcentage_haut", "largeur_voisinage", "seuil_decallage", "r_max",
                "supprimer_continum", "facteur_continum", "supprimer_mediane", "largeur_exclusion"]:
        print(f"    {nom} = {courant[nom]}")

    print(f"\n  départ    : score={score_depart:6.3f} (A={A_depart:.2f}, retention={retention_depart:.2f}, d={d_depart:.2f})")
    print(f"  calibré   : score={score_final:6.3f} (A={A_final:.2f}, retention={retention_final:.2f}, d={d_final:.2f})")
    print("-" * 100)

    configuration = dict(courant)
    configuration["tab_kp"] = TAB_KP
    configuration["tab_vsys"] = TAB_VSYS
    configuration["seuil_decallage"] = SEUIL_DECALLAGE
    configuration["propagation_erreurs"] = PROPAGATION_ERREURS
    configuration["elargissement_rotationnel"] = ELARGISSEMENT_ROTATIONNEL
    configuration["abondance"] = ABONDANCE

    return configuration
