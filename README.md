



# Versatile Diffusion Model


![Exemple de sortie](./assets/front.png)


## Introduction
<p align="justify">
L'intelligence artificielle générative a connu un essor spectaculaire ces dernières années, notamment avec les progrès réalisés dans les modèles capables de générer des images, du texte, de la musique et d'autres types de contenus à partir de données d'entrée. Parmi ces modèles, les modèles de diffusion ont émergé comme l'une des approches les plus prometteuses pour la génération de contenu visuel. Ces modèles ont prouvé leur efficacité dans la création d'images réalistes à partir de bruit aléatoire en inversant progressivement un processus de diffusion. Cela permet de générer des images de haute qualité à partir de descriptions textuelles ou de bruit aléatoire.
Les modèles de diffusion classiques, tels que Stable Diffusion ou DALL-E, sont principalement spécialisés dans une tâche donnée, soit la génération d'images à partir de texte ou la création d'images réalistes à partir de bruit. Cependant, ces modèles, bien que puissants, sont souvent limités dans leur capacité à traiter plusieurs modalités de manière fluide. Cela signifie qu'ils ne peuvent pas simultanément gérer plusieurs types de données, comme des images et des textes, dans une approche flexible. C'est ici qu'intervient le modèle Versatile Diffusion (VD), qui propose une approche intégrée et flexible pour gérer à la fois des images et du texte dans un seul modèle.
</p>



<p align="center">
  <img src="./assets/model.png" />
</p>

<p align="center">
  Figure 1 Schéma du modèle versatile utilisé dans la revue
</p>
<br>


## Une approche unifiée pour le texte et l'image


<p align="justify">
Le modèle Versatile Diffusion représente une avancée significative par rapport aux modèles de diffusion traditionnels en ce qu'il est capable de gérer des tâches multimodales de manière fluide. Contrairement aux modèles de diffusion classiques, qui se concentrent généralement sur une seule modalité, VD permet de traiter à la fois des images et des descriptions textuelles au sein d'un même modèle. Cela signifie qu'il est capable de générer des images à partir de descriptions textuelles, mais aussi de générer des légendes ou des descriptions à partir d'images. De plus, il permet des modifications d'images basées sur des instructions textuelles, offrant ainsi une grande flexibilité et de nouvelles possibilités pour l'interaction avec des contenus multimodaux.
Une caractéristique clé de Versatile Diffusion est sa capacité à effectuer une diffusion flexible dans plusieurs directions, que ce soit pour générer du texte à partir d'images, des images à partir de texte ou encore des variations d'images selon des spécifications textuelles. Cela permet d'exploiter les relations entre le texte et l'image d'une manière plus fluide et plus cohérente que les modèles traditionnels, qui traitent souvent ces deux types de données de manière séparée.

</p>



<p align="center">
  <img src="./assets/cars.png" />
</p>

<p align="center">
  Figure 2 Exemple de resultats "image to image"
</p>
<br>


## Objectif
<p align="justify">
Dans le cadre de ce projet, nous allons implémenter le modèle Versatile Diffusion et l'évaluer. L'objectif principal est d’analyser ses capacités à traiter des tâches multimodales en générant des images à partir de texte, en générant du texte à partir d'images et en modifiant des images sur la base de descriptions textuelles.
Nous évaluerons également la flexibilité du modèle, notamment sa capacité à générer des variations d'images à partir de descriptions textuelles et à séparer le style du contenu. En outre, nous explorerons les applications pratiques du modèle, telles que la génération de contenu créatif, la recherche visuelle ou la création automatique de légendes pour des images.
</p>
<br>

## Références
<p align="justify">
Xu, Xingqian, et al. "Versatile Diffusion: Text, Images and Variations All in One Diffusion Model." arXiv, 2022, arxiv.org/abs/2211.08332.
</p>




























il faut télécharger les models => https://huggingface.co/shi-labs/versatile-diffusion/tree/main/pretrained_pth
