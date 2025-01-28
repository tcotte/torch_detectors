# Alternatives de YOLOv8 Ultralytics

### YOLOv8 Keras - Tensorflow
A la base, l'objectif de ce projet était de pouvoir remplacer le modèle YOLOv8-n d'Ultralytics par
une architecture YOLOv8 proposée par Keras. Nous avons essayé plusieurs fois d'implémenter ce modèle 
grâce à l'environnement de Keras. Nous pouvons faire quelques reproches à cet environnement:
- Environnement assez fermé: il est assez compliqué de changer le code car beaucoup de fonctions machent
le travail. Cependant, la librarie donne offre peu de choix sur certains paramètres (comme les 
augmentations de données). Nous avons remarqué que nous pouvions élargir l'éventail en créant nos propres
augmentations au format Pytorch mais nous n'avons pas encore essayé.
- Difficile d'utilisation du GPU sous Windows. En effet, depuis la version 2.10, Tensorflow ne supporte 
plus l'utilisation du GPU sous Windows. Or, la version de Keras_CV qui comporte l'architecture YOLOv8
nécessite une version de Tensorflow supérieure à 2.10. Nous avons donc essayé de créer un environnement
Python sous WSL comportant une version de Tensorflow qui supporte l'utilisation du GPU sous Linux. Cet 
envrironnement fonctionne bien sous Pycharm lors d'une exécution "normal" mais ne fonctionne pas lorsque
nous utilisons le débugger.
- Taille de l'image ??? 

Nous avons réussi à entrainer le model et nous obtenons les résultats suivants: 



Nous avons profité de l'entraînement de ce model pour pouvoir implémenter les *loggers* de Picsellia
(=*experiment*). On peut retrouver une *experiment* ici: https://app.picsellia.com/0191ff2c-46e2-75d3-92a3-eade82ae7183/project/01944ab2-4f8a-707f-9b8d-58e85cec6465/experiment/01944ab4-4ac4-71af-8319-3f440755d7a2/.


Pour palier aux deux premiers inconvénients de Keras, nous avons essayé d'implémenter le model sous 
Pytorch. Nous avons trouvé un tutoriel[1] écrit par Keras qui explique comment entraîner un model Keras
avec un backend Pytorch. Nous avons essayé de transposer ce script à YOLOv8 
(*src/torch_implementation/model.py*) mais il semblerait que ce soit impossible vu que le model est 
sous Keras_CV et non pas Keras (il utilise des tenseurs venant de Tensorflow). Le projet a donc été 
abandonné.

Cependant, puisque nous avions commencé à implémenté le un model d'object detection sous Pytorch, nous
avons regardé les models dont dispose la librairie et nous avons essayé d'en implémenté pour voir si nous
avons des résultats intéressants avec ces models. 


### Idées

Implementer Nanodet: https://github.com/RangiLyu/nanodet


### RetinaNet

TODO:
- Envoyer les paramètres de normalization dans le model + parametres de post-process https://pytorch.org/vision/0.13/models/generated/torchvision.models.detection.retinanet_resnet50_fpn_v2.html#torchvision.models.detection.retinanet_resnet50_fpn_v2
- remove A.Normalize() from transform and pass std/mean parameters to model
- [To test] change number of channels to freeze 
- Ajouter la possibilité d'inclure des poids déjà entraînés du backbone
- [To test] Add patience
- [DONE] Add possibility to see validation loss https://discuss.pytorch.org/t/compute-validation-loss-for-faster-rcnn/62333
- learning rate decay https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
- [DONE] implement learning rate warmup -> only exponential warmup implemented
- Ré-essayer la normalization avec les params calculés en mode 'custom dataset'
- Transform model to ONNX
- Try grid evaluation with SGD (-> different learning rates)
- [DONE]Import custom torch weights
- Import backbone weights
- Include mosaic and other augmentations from YOLO

- [DONE]Display precision_recall curve
- [test]Print available space on CUDA
- [test]Import existing weights 
- Création evaluation experiment picsellia
- 
- 



### Bibliography

[1] https://keras.io/guides/writing_a_custom_training_loop_in_torch/
