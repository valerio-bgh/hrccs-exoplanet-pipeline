"""
Fonctions d'affichage : génère et sauvegarde en PDF (dans result/) toutes les figures de la pipeline : spectres temporels
bruts/nettoyes/masques, SNR, airmass, transit, evolution de la PCA selon r, CCF, cartes de significativite Kp-Vsys et leurs
tranches au Kp reel.
"""

import matplotlib.pyplot as plt
import numpy as np

# Palette d'affichage génerique
palette = plt.get_cmap('viridis').copy()
palette.set_bad(color='red')

# Palette divergente pour les quantites signées centrees sur 0 de faible variance (z-score de la carte Kp-Vsys)
palette_signee = plt.get_cmap('RdBu_r').copy()
palette_signee.set_bad(color='lime')



def afficher_tableau_3D(data, wavelength, phases, nom_fichier, titre = ""):
    """
    Affiche chaque ordre comme un spectre temps (axe y) * longueur d'onde (axe x)

    Entrées :
        data        -> de la forme (n_ordres, n_temps, n_pixel)
        wavelength  -> abscisse des images, tableau 2D de la forme (n_ordre, n_pixel)
        phase       -> tableau 1D (n_temps,), sans unité mais fraction d'une periode orbitale
        nom_fichier -> nom du PDF en sortie
        titre       -> (optionnel) titre de la figure
    """
    n_ordre = data.shape[0]
    fig, ordre = plt.subplots(nrows=n_ordre, ncols=1, figsize=(10, 2.5 * n_ordre))
    if n_ordre == 1:
        ordre = [ordre]

    for n in range(n_ordre):
        frame = data[n]
        n_temps = frame.shape[0]

        # Si tout les floats du tableau sont non fini (donc nan ou infini) ou si le maximum est nul (sert à gerer les ordres non calibrés)
        if np.all(~np.isfinite(frame)) or np.nanmax(frame) == 0:
            ordre[n].text(0.5, 0.5, "Ordre invalide (non calibré)", ha="center", va="center", transform=ordre[n].transAxes)
            ordre[n].set_title(f"Ordre {n}")
            
        else:
            image = ordre[n].imshow(frame, cmap=palette, vmin=np.nanpercentile(frame, 1), vmax=np.nanpercentile(frame, 99), aspect='auto', origin='lower', extent=[wavelength[n][0], wavelength[n][-1], phases[0], phases[-1]])
            ordre[n].set_title(f"Ordre {n} - {titre}")
            ordre[n].set_xlabel("Longueur d'onde (nm)")
            ordre[n].set_ylabel("Phase")
            fig.colorbar(image, ax=ordre[n]) 


    plt.tight_layout() # Gère la separation entre les spectres
    plt.savefig(f"result/{nom_fichier}.pdf", dpi=300)
    plt.close(fig)



def afficher_snr(snr, phases, nom_fichier):
    """
    Trace le snr en fonction du temps, une courbe = un ordre

    Entrée :
        snr         -> tableau 2D (n_ordres, n_temps)
        phase       -> tableau 1D (n_temps,), sans unité mais fraction d'une periode orbitale
        nom_fichier -> nom du pdf de sortie
    """
    n_ordres, n_temps = snr.shape

    fig, ax = plt.subplots(figsize=(10, 6))

    for n in range(n_ordres):
        ligne = snr[n]
        if not (np.all(~np.isfinite(ligne)) or np.nanmax(ligne) == 0):
            ax.plot(phases, ligne, marker='o', label=f"Ordre {n}")

    ax.set_xlabel("Phase orbitale")
    ax.set_ylabel("SNR")
    ax.set_title("Évolution du rapport signal/bruit par ordre")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"result/{nom_fichier}.pdf", dpi=300)
    plt.close(fig)



def afficher_airmass(airmass, phases, nom_fichier):
    """
    Trace la masse d'air en fonction de la phase orbitale

    Entrée :
        airmass     -> tableau 1D (n_temps,)
        phases      -> tableau 1D (n_temps,), sans unité mais fraction d'une periode orbitale
        nom_fichier -> nom du pdf de sortie
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(phases, airmass, marker='o', ms=3)

    ax.set_xlabel("Phase orbitale")
    ax.set_ylabel("Airmass")
    ax.set_title("Évolution de la masse d'air au cours de l'observation")

    plt.tight_layout()
    plt.savefig(f"result/{nom_fichier}.pdf", dpi=300)
    plt.close(fig)



def afficher_tell(tell, wavelength, nom_fichier):
    """
    Affiche pour chaque ordre les raies tellurique sous forme d'image (longueur d'onde en x)

    Entrées :
        tell.       -> tableau 2D (n_ordres, n_pixels)
        wavelength  -> tableau 2D (n_ordres, n_pixels) de l'abscisse 
        nom_fichier -> nom du PDF de sortie
    """
    n_ordre = tell.shape[0]
    fig, axes = plt.subplots(nrows=n_ordre, ncols=1, figsize=(10, 1.8 * n_ordre))
    if n_ordre == 1:
        axes = [axes]

    for n in range(n_ordre):
        ligne = tell[n]
        ax = axes[n]

        if np.all(~np.isfinite(ligne)) or np.nanmax(ligne) == 0:
            ax.text(0.5, 0.5, "Ordre invalide (non calibré)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Ordre {n}")

        else:
            frame = np.tile(ligne, (50, 1))

            image = ax.imshow(frame, cmap=palette, vmin=np.nanmin(ligne), vmax=np.nanmax(ligne), aspect='auto', origin='lower', extent=[wavelength[n][0], wavelength[n][-1], 0, 50])
            ax.set_title(f"Ordre {n} — erreur")
            ax.set_xlabel("Longueur d'onde (nm)")
            ax.set_yticks([])
            fig.colorbar(image, ax=ax)

    plt.tight_layout()
    plt.savefig(f"result/{nom_fichier}.pdf", dpi=300)
    plt.close(fig)



def afficher_transit(poids_transit, en_transit, phases, nom_fichier):
    """
    Trace le poids de transit normalisé en fonction de la phase orbitale, colore les poses en transit différemment de la baseline

    Entrée :
        poids_transit -> tableau 1D (n_temps,), poids normalisé du transit
        en_transit    -> tableau booléen 1D (n_temps,), True = en transit
        phases        -> tableau 1D (n_temps,), sans unité mais fraction d'une periode orbitale
        nom_fichier   -> nom du pdf de sortie (sans extension)
    """
    fig, ax = plt.subplots(figsize=(10, 4)) 

    ax.plot(phases, poids_transit, color='0.6', lw=1, zorder=1)
    ax.scatter(phases[~en_transit], poids_transit[~en_transit], color='tab:blue', s=20, label="baseline", zorder=2)
    ax.scatter(phases[en_transit], poids_transit[en_transit], color='tab:red', s=20, label="transit", zorder=2)
    ax.axhline(0.0, color='0.8', ls='--', lw=0.8, zorder=0)  # repere : poids nul hors transit
    ax.set_ylabel("Poids de transit")
    ax.legend()

    ax.set_xlabel("Phase orbitale")
    ax.set_title("Poids de transit et état de chaque observation")

    plt.tight_layout()
    plt.savefig(f"result/{nom_fichier}.pdf", dpi=300)
    plt.close(fig)



def afficher_evolution_pca(data_4D, wavelength, phases, nom_dossier, element):
    """
    Pour chaque ordre, affiche une colonne de r_max+1 images empilées verticalement : r=0 en haut, r=r_max en bas

    Entrées :
        data_4D     -> tableau 4D (n_ordres, n_temps, n_pixels, r_max+1) où data_4D[..., r] = résultat PCA avec r composantes retirées
        wavelength  -> abscisses, tableau 2D (n_ordres, n_pixels)
        phases      -> tableau 1D (n_temps,), phases orbitales, ordonnée
        nom_dossier -> nom du dossier ou stocker les figures
        element     -> molécule traitée : sous-dossier de Evolution_PCA, pour ne pas
                       écraser les figures d'une molécule avec celles d'une autre sur le même fichier
    """
    n_ordres, n_temps, n_pixels, n_r = data_4D.shape

    for n in range(n_ordres):
        spectre, ordre = plt.subplots(nrows=n_r, ncols=1, figsize=(10, 2.2 * n_r))
        if n_r == 1:                      # si une seule valeur de r, ordre n'est pas un tableau
            ordre = [ordre]

        for r in range(n_r):
            frame = data_4D[n, :, :, r]

            if np.all(~np.isfinite(frame)) or np.nanmax(frame) == 0:
                ordre[r].text(0.5, 0.5, "Ordre invalide (non calibré)", ha="center", va="center", transform=ordre[r].transAxes)
                ordre[r].set_title(f"Ordre {n} — r = {r}")

            else:
                image = ordre[r].imshow(frame, cmap=palette, vmin=np.nanpercentile(frame, 1), vmax = np.nanpercentile(frame, 99), aspect='auto', origin='lower', extent = [wavelength[n][0], wavelength[n][-1], phases[0], phases[-1]]) # aspect='auto' donne la forme d'origine non déformé et origin='lower' que en monttant sur l'axe y on avance dans le temps, et extent = [x_min, x_max, y_min, y_max] 
                ordre[r].set_title(f"Ordre {n} pour r={r}")
                ordre[r].set_ylabel("Phase")
                if r == n_r - 1:              # etiquette abscisse seulement sur l'image du bas
                    ordre[r].set_xlabel("Longueur d'onde (nm)")
                spectre.colorbar(image, ax=ordre[r])
        plt.tight_layout()
        plt.savefig(f"result/{nom_dossier}/Evolution_PCA/{element}/ordre{n}.pdf", dpi=300)
        plt.close(spectre)



def afficher_ccf(CCF, tab_vitesse, phases, nom_fichier):
    """
    Affiche la carte de corrélation croisée CCF(temps, vitesse) sous forme d'image : vitesse d'essai en x, phase orbitale en y

    Entrées :
        CCF         -> tableau 2D (n_temps, n_v)
        tab_vitesse -> tableau 1D (n_v,), les vitesses d'essai (km/s), abscisse
        phases      -> tableau 1D (n_temps,), les phases orbitales, ordonnée
        nom_fichier -> nom du PDF de sortie
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    if np.all(~np.isfinite(CCF)):
        ax.text(0.5, 0.5, "CCF invalide (non calibrée)", ha="center", va="center", transform=ax.transAxes)

    else:
        image = ax.imshow(CCF, cmap=palette, aspect='auto', origin='lower', extent=[tab_vitesse[0], tab_vitesse[-1], phases.min(), phases.max()], vmin=np.nanpercentile(CCF, 1), vmax=np.nanpercentile(CCF, 99))

        ax.set_xlabel("Vitesse d'essai (km/s)")
        ax.set_ylabel("Phase orbitale")
        ax.set_title("Corrélation croisée CCF(t, v)")
        fig.colorbar(image, ax=ax, label="R(t, v)")

    plt.tight_layout()
    plt.savefig(f"result/{nom_fichier}.pdf", dpi=300)
    plt.close(fig)



def ticks_avec_pic(ticks_defaut, valeur_pic, borne_min, borne_max):
    """
    Programme optionnel pour améliorer la lisibilité  des etiquettes sur les axes pour évité la superposition des données 

    Entrée :
        ticks_defaut          -> ticks proposés par matplotlib
        valeur_pic            -> position du maximum, à marquer obligatoirement
        borne_min, borne_max  -> limites de l'axe, les ticks hors bornes sont ecartés

    Sortie :
        ticks -> tableau 1D trié, contenant valeur_pic et les ticks par défaut suffisamment éloignés
    """
    ticks = np.asarray(ticks_defaut, dtype=float)
    ticks = ticks[(ticks >= borne_min) & (ticks <= borne_max)]

    if len(ticks) >= 2:
        ecart_min = 0.6 * np.min(np.diff(np.sort(ticks)))   # 60 % du pas des ticks : place suffisante pour l'étiquette
    else:
        ecart_min = 0.0

    ticks = ticks[np.abs(ticks - valeur_pic) > ecart_min]

    return np.sort(np.append(ticks, valeur_pic))



def afficher_carte_kp_vsys(carte_sigma, tab_kp, tab_vsys, nom_fichier, kp_attendu=None):
    """
    Affiche la carte Kp-Vsys en z-score, Vsys en x, Kp en y, trace deux repères :
        - une croix VERTE au maximum global de la carte
        - deux lignes NOIRES qui se croisent au Vsys maximisant la ligne Kp = kp_attendu

    Entrée :
        carte_sigma      -> (n_kp, n_vsys)
        tab_kp, tab_vsys -> grilles d'essai
        nom_fichier      -> nom du PDF en sortie
        kp_attendu       -> (optionnel) Kp imposé par les données orbitales en km/s, si None les lignes se croisent sur le maximum global comme la croix
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if np.all(~np.isfinite(carte_sigma)):
        ax.text(0.5, 0.5, "Carte invalide (non calibrée)", ha="center", va="center", transform=ax.transAxes)

    else:
        borne = np.nanmax(np.abs([np.nanpercentile(carte_sigma, 1), np.nanpercentile(carte_sigma, 99)]))
        image = ax.imshow(carte_sigma, cmap=palette_signee, aspect='auto', origin='lower', extent=[tab_vsys[0], tab_vsys[-1], tab_kp[0], tab_kp[-1]], vmin=-borne, vmax=borne)

        # Maximum global de la carte -> croix verte (le vert reste lisible sur toute la palette rouge/blanc/bleu)
        ikp, ivs = np.unravel_index(np.nanargmax(carte_sigma), carte_sigma.shape)
        ax.plot(tab_vsys[ivs], tab_kp[ikp], '+', color='limegreen', ms=12, mew=2.0,
                label=f"max carte : Kp={tab_kp[ikp]:.2f}, Vsys={tab_vsys[ivs]:.2f} ({carte_sigma[ikp, ivs]:.2f} σ)")

        # Lignes noires : Kp imposé par l'orbite, et Vsys qui maximise CETTE ligne uniquement
        if kp_attendu is None:
            ikp_ligne, ivs_ligne = ikp, ivs
        else:
            ikp_ligne = int(np.argmin(np.abs(np.asarray(tab_kp) - kp_attendu)))
            ivs_ligne = int(np.nanargmax(carte_sigma[ikp_ligne]))
            ax.plot(tab_vsys[ivs_ligne], tab_kp[ikp_ligne], 'x', color='black', ms=9, mew=1.8, label=f"Kp orbital : Kp={tab_kp[ikp_ligne]:.2f}, Vsys={tab_vsys[ivs_ligne]:.2f} ({carte_sigma[ikp_ligne, ivs_ligne]:.2f} σ)")
        ax.axhline(tab_kp[ikp_ligne], color='black', ls='--', lw=0.8, alpha=0.7)
        ax.axvline(tab_vsys[ivs_ligne], color='black', ls='--', lw=0.8, alpha=0.7)

        ax.set_xlim(tab_vsys[0], tab_vsys[-1])
        ax.set_ylim(tab_kp[0], tab_kp[-1])
        ax.set_xticks(ticks_avec_pic(ax.get_xticks(), tab_vsys[ivs_ligne], tab_vsys[0], tab_vsys[-1]))
        ax.set_yticks(ticks_avec_pic(ax.get_yticks(), tab_kp[ikp_ligne], tab_kp[0], tab_kp[-1]))
        ax.legend(loc='upper right', fontsize=7, framealpha=0.8)

        ax.set_xlabel("Vsys d'essai (km/s)")
        ax.set_ylabel("Kp d'essai (km/s)")
        ax.set_title("Carte de significativité")
        fig.colorbar(image, ax=ax, label="Significativité (σ)")

    plt.tight_layout()
    plt.savefig(f"result/{nom_fichier}.pdf", dpi=300)
    plt.close(fig)



def afficher_tranche_kp(carte_sigma, tab_kp, tab_vsys, kp_reel, nom_fichier):
    """
    Trace une tranche horizontale de la carte Kp-Vsys au Kp réel, la significativité (z-score) en fonction de Vsys et indique le maximum

    Entrée :
        carte_sigma      -> (n_kp, n_vsys)
        tab_kp, tab_vsys -> grilles d'essai
        kp_reel          -> Kp connu de la planète (km/s), sélectionne la ligne la plus proche dans tab_kp
        nom_fichier      -> nom du PDF en sortie
    """
    ikp = int(np.argmin(np.abs(np.asarray(tab_kp) - kp_reel)))
    tranche = carte_sigma[ikp, :]

    fig, ax = plt.subplots(figsize=(9, 3.5))

    if np.all(~np.isfinite(tranche)):
        ax.text(0.5, 0.5, "Tranche invalide (non calibrée)", ha="center", va="center", transform=ax.transAxes)

    else:
        ax.plot(tab_vsys, tranche)
        ax.axhline(0, color='0.8', ls='--', lw=0.8)

        # Maximum de la tranche -> lignes pointillées rouges sur son abscisse (Vsys) et son ordonnée (sigma)
        ivs_max = int(np.nanargmax(tranche))
        vsys_max, sigma_max = tab_vsys[ivs_max], tranche[ivs_max]
        ax.axvline(vsys_max, color='tab:red', ls='--', lw=0.8, alpha=0.7)
        ax.axhline(sigma_max, color='tab:red', ls='--', lw=0.8, alpha=0.7, label=f"max : Vsys={vsys_max:.2f} km/s ({sigma_max:.2f} σ)")

        ax.set_xlim(tab_vsys[0], tab_vsys[-1])
        ax.set_xticks(ticks_avec_pic(ax.get_xticks(), vsys_max, tab_vsys[0], tab_vsys[-1]))
        ax.set_yticks(ticks_avec_pic(ax.get_yticks(), sigma_max, np.nanmin(tranche), np.nanmax(tranche)))

        ax.set_xlabel("Vsys d'essai (km/s)")
        ax.set_ylabel("Significativité (σ)")
        ax.set_title(f"Tranche à Kp={tab_kp[ikp]:.2f} km/s")

    plt.tight_layout()
    plt.savefig(f"result/{nom_fichier}.pdf", dpi=300)
    plt.close(fig)



def afficher_zscore_abondance(abondances, zscores, seuil_sigma, abondance_limite, nom_fichier):
    """
    Trace la significativité (z-score) au point injecté en fonction de l'abondance testée (échelle log)

    Entrée :
        abondances       -> liste/tableau des abondances testées (fraction massique)
        zscores           -> liste/tableau des z-score correspondants, au même indice
        seuil_sigma       -> seuil de détection utilisé
        abondance_limite  -> abondance limite retenue par la dichotomie (fraction massique)
        nom_fichier       -> nom du PDF en sortie
    """
    abondances = np.asarray(abondances)
    zscores = np.asarray(zscores)
    ordre = np.argsort(abondances)
    abondances, zscores = abondances[ordre], zscores[ordre]

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(abondances) == 0 or np.all(~np.isfinite(zscores)):
        ax.text(0.5, 0.5, "Courbe invalide (aucun point valide)", ha="center", va="center", transform=ax.transAxes)

    else:
        ax.plot(abondances, zscores, 'o-', color='tab:blue', ms=4, label="z-score au point injecté")
        ax.axhline(seuil_sigma, color='tab:red', ls='--', lw=1.0, label=f"seuil = {seuil_sigma:.1f} σ")
        ax.axvline(abondance_limite, color='tab:red', ls='--', lw=1.0, label=f"limite = {abondance_limite:.2e}")

        ax.set_xscale('log')
        ax.set_xlabel("Abondance testée (fraction massique)")
        ax.set_ylabel("Significativité au point injecté (σ)")
        ax.set_title("Sensibilité de détection en fonction de l'abondance")
        ax.legend(loc='best', fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(f"result/{nom_fichier}.pdf", dpi=300)
    plt.close(fig)
