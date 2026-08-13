"""
Fonctions utilitaires d'entree/sortie : ouverture des fichiers .pickle de donnees, creation des arborescences de dossiers dans
result/ (une par etape/dossier de sortie), et ecriture de la configuration utilisee dans un fichier texte recapitulatif.
"""

import os
import pickle as pic



def ouvrir(fichier):
    """
    Peremt d'ouvrir le fichier .pickel, renvoie les donnée

    Entrée :
        fichier : (string) nom du fichier à ouvrir sans l'extention .pickel

    Sortie :
        data : données contenues dans le .pickel
    """
    with open(f"data/{fichier}.pickle", "rb") as donnee:
        data = pic.load(donnee)
    return data



def gestion_dossiers(nom_sortie, element):
    """
    Vérifie que result/{nom_sortie}/ existe (le crée sinon), ainsi que les sous-dossiers utilisés par le pipeline (CCF, Evolution_PCA/{element}, Carte, Configurations, Tranche_carte)
    et crée ceux qui manquent

    Entrée :
        nom_sortie -> nom du sous-dossier de result/ ("fichier" dans main.py)
        element    -> molécule traitée, determine le sous-dossier de Evolution_PCA
    """
    sous_dossiers = ["CCF", f"Evolution_PCA/{element}", "Carte", "Configurations", "Tranche_carte"]
    for sous_dossier in sous_dossiers:
        os.makedirs(f"result/{nom_sortie}/{sous_dossier}", exist_ok=True)



def gestion_dossiers_injection(nom_sortie):
    """
    Vérifie que result/{nom_sortie}/Injection/ existe (le crée sinon), ainsi que les sous-dossiers utilisés par test_injection.py

    Entrée :
        nom_sortie -> nom du sous-dossier de result/ ("fichier" dans test_injection.py)

    Sortie :
        aucune
    """
    sous_dossiers = ["CCF", "Carte", "Tranche_carte", "Configurations"]
    for sous_dossier in sous_dossiers:
        os.makedirs(f"result/{nom_sortie}/Injection/{sous_dossier}", exist_ok=True)



def gestion_dossiers_limite(nom_sortie):
    """
    Vérifie que result/{nom_sortie}/Limite_abondance/ existe (le crée sinon), ainsi que les sous-dossiers utilisés par limite_abondance.py

    Entrée :
        nom_sortie -> nom du sous-dossier de result/ ("fichier" dans limite_abondance.py)

    Sortie :
        aucune
    """
    sous_dossiers = ["Carte", "Tranche_carte", "Zscore", "Configurations"]
    for sous_dossier in sous_dossiers:
        os.makedirs(f"result/{nom_sortie}/Limite_abondance/{sous_dossier}", exist_ok=True)



def gestion_dossier_mixte(dossier):
    """
    Vérifie que result/{dossier}/ existe (le crée sinon), ainsi que les sous-dossiers Carte et Tranche_carte

    Entrée :
        dossier -> nom du sous-dossier de result/ dédié à la carte fusionnée
    """
    for sous_dossier in ["Carte", "Tranche_carte"]:
        os.makedirs(f"result/{dossier}/{sous_dossier}", exist_ok=True)



def ecrire_configuration(configuration, parametres_appel, chemin):
    """
    Écrit la configuration utilisée dans un fichier texte, et commente chaque paramètre

    Entrée :
        configuration    -> dict de paramètres
        parametres_appel -> dict {nom du paramètre : valeur}, paramètres d'appel de pipeline()/test_injection() à inscrire en bas du fichier
        chemin           -> chemin complet du fichier texte (écrire avec extension)
    """
    def arange_txt(valeur): # Gère les tableaux pour éviter de l'imprimer en entier
        pas = valeur[1] - valeur[0] if len(valeur) > 1 else 1
        return f"np.arange({valeur[0]}, {valeur[-1] + pas}, {pas})" 

    tab_kp = f"tab_kp = {arange_txt(configuration['tab_kp'])}"
    tab_vsys = f"tab_vsys = {arange_txt(configuration['tab_vsys'])}"
    seuil_decallage = f"seuil_decallage = {configuration['seuil_decallage']}"
    propagation_erreurs = f"propagation_erreurs = {configuration['propagation_erreurs']}"
    elargissement_rotationnel = f"elargissement_rotationnel = {configuration['elargissement_rotationnel']}"
    abondance = f"abondance = {configuration['abondance']}"
    pourcentage_bas = f"pourcentage_bas = {configuration['pourcentage_bas']}"
    pourcentage_haut = f"pourcentage_haut = {configuration['pourcentage_haut']}"
    largeur_voisinage = f"largeur_voisinage = {configuration['largeur_voisinage']}"
    r_max = f"r_max = {configuration['r_max']}"
    supprimer_continum = f"supprimer_continum = {configuration['supprimer_continum']}"
    facteur_continum = f"facteur_continum = {configuration['facteur_continum']}"
    supprimer_mediane = f"supprimer_mediane = {configuration['supprimer_mediane']}"
    largeur_exclusion = f"largeur_exclusion = {configuration['largeur_exclusion']}"

    L = 40  # largeur de la colonne avant le commentaire
    texte = (
        "Configuration utilisée :\n\n"
        f"    {tab_kp:<{L}}# en km/s, grille d'ecarts autour du Kp connu\n"
        f"    {tab_vsys:<{L}}# en km/s, grille d'ecarts autour du Vsys connu\n"
        f"    {seuil_decallage:<{L}}# masque_decallage : masque les raies qui se décalent entre les prises A et B\n"
        f"    {propagation_erreurs:<{L}}# propager_erreur : propage rigoureusement l'effet de la PCA sur les erreurs recues dans le .pickel\n"
        f"    {elargissement_rotationnel:<{L}}# spectre_pRT_to_CRIRES : applique l'elargissement rotationnel au template (profil de Gray, rotation supposee synchrone)\n"
        f"    {abondance:<{L}}# spectre_transmission : fraction massique (mass_fraction) de la molecule recherchee dans l'atmosphere\n"
        "\n"
        f"    {pourcentage_bas:<{L}}# masquage tellurique : profondeur minimale masquee\n"
        f"    {pourcentage_haut:<{L}}# masquage tellurique : profondeur maximale masquee\n"
        f"    {largeur_voisinage:<{L}}# masquage tellurique : largeur du voisinage etendu autour des raies profondes\n"
        f"    {r_max:<{L}}# pca_3D_to_4D : nombre de composantes PCA retirees (bruit/tellurique residuel)\n"
        f"    {supprimer_continum:<{L}}# spectre_pRT_to_CRIRES : retire le continuum du template avant correlation\n"
        f"    {facteur_continum:<{L}}# spectre_pRT_to_CRIRES : largeur de la convolution utilisee pour retirer le continuum\n"
        f"    {supprimer_mediane:<{L}}# correlation_croisee : retire la mediane de la CCF (par vitesse) avant de sommer\n"
        f"    {largeur_exclusion:<{L}}# somme_to_zscore : demi-largeur exclue autour du Vsys testé pour estimer le bruit local\n"
    )


    texte += "\nParamètres d'appel :\n\n"
    for nom, valeur in parametres_appel.items():
        texte += f"    {nom} = {valeur}\n"

    with open(chemin, "w") as f:
        f.write(texte)
