# 1. Use the server

1. push votre programme sur higgs_challenge github
2. se rendre sur votre terminal
3. connexion ssh via la commande suivante

`ssh other@93.31.104.97 -p 2021`

Pour avoir un retour écran sur votre ordinateur :

`ssh –X other@93.31.104.97 -p 2021`

4. renseigner le mot de passe
5. Si vous n'avez pas fait d'erreur, vous y êtes.

Veuillez vérifier les « écrans » ouverts avec la commande suivante

`screen –ls`

Un écran permet de faire tourner vos propres processus et de 
pouvoir vous déconnecter sans interrompre ces processus. 

6. Si votre écran n’est pas présent, créer le en remplaçant dans 
la commande suivante nom\_de\_votre\_ecran par le nom de votre ecran 
en respectant les conventions:

`screen -S nom_de_votre_ecran`

7. rendez-vous sur votre écran en remplaçant dans 
la commande suivante nom\_de\_votre\_ecran par le nom de votre ecran:

`screen -rd nom_de_votre_ecran`

Vous pouvez quitter cet ecran **SANS** interrompre les processus en cours en appuyant 
simultanément sur `Ctrl` + `a` + `d`

Vous pouvez afficher les ecrans disponibles via la commande:

`screen –ls`

Vous pouvez retourner sur votre écran via la commande:

`screen -rd nom_de_votre_ecran`

8. *(facultatif)* voici quelques commandes utiles :

* Ne pas hésiter à pull la dernière version

`git pull`

* Si erreur du type : "ImportError: No module named ..."

`export PYTHONPATH=$PYTHONPATH:/home/dray/Documents/higgs_challenge`

### Attention:
Lors du re-démarrage du server, les écrans et donc les processus  sont perdus.


# Jupyter running on the server
1. Se connecter au server

2. Lancer cette commande dans le terminal qui tourne sur le server via un ecran dédié à Jupyter avec la commande `screen`, puis lancez Jupyter.

`jupyter notebook --no-browser --port=2021`

Un **token d'authentification** sera affiché, copiez le.

3. Ouvrir un autre terminal et lancer la commande suivante

`ssh -N -f -L 127.0.0.1:2021:127.0.0.1:2021 other@93.31.104.97 -p 2021`

4. Dans un navigateur sur votre machine, rendez-vous à cette adresse:

`http://127.0.0.1:2021`

Vous n'aurez plus qu'à y coller votre token ! Bonne utilisation !

#### source : [How to run Jupyter from remote machine](https://kawahara.ca/how-to-run-an-ipythonjupyter-notebook-on-a-remote-machine/)
