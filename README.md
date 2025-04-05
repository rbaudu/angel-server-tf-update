# Angel Server TensorFlow Update

Ce dépôt contient les classes Java mises à jour pour utiliser TensorFlow 0.5.0 dans le projet angel-server-capture.

## Classes mises à jour

1. **VideoUtils.java**
   - Conversion d'images OpenCV en tenseurs TensorFlow
   - Support pour les formats uint8 et float32
   - Méthodes améliorées de prétraitement d'images

2. **ModelLoader.java**
   - Chargement optimisé des modèles TensorFlow
   - Informations de diagnostic améliorées

3. **PresenceDetector.java**
   - Détection de présence humaine optimisée
   - Utilisation de l'API TensorFlow moderne
   - Meilleur support de débogage

4. **VisualActivityClassifier.java**
   - Classification d'activités avec TensorFlow
   - Gestion optimisée des tenseurs d'entrée et de sortie

5. **TensorFlowDiagnostics.java** (Nouvelle classe)
   - Outils de diagnostic pour les modèles TensorFlow
   - Analyse des opérations et tenseurs
   - Détection automatique des besoins de normalisation

## Comment utiliser ces mises à jour

1. Assurez-vous que votre fichier pom.xml inclut la dépendance TensorFlow 0.5.0 :
   ```xml
   <dependency>
       <groupId>org.tensorflow</groupId>
       <artifactId>tensorflow-core-platform</artifactId>
       <version>0.5.0</version>
   </dependency>
   ```

2. Remplacez les classes existantes par celles fournies dans ce dépôt.

3. Pour déboguer les problèmes liés à TensorFlow, utilisez la nouvelle classe `TensorFlowDiagnostics`.

## Principales améliorations

- **Normalisation des images** : Support correct pour les modèles qui attendent des entrées normalisées (0-1) ou non normalisées (0-255)
- **Gestion des tenseurs** : Utilisation de l'API moderne pour manipuler les tenseurs
- **Performances** : Optimisation des conversions d'images
- **Débogage** : Outils avancés pour identifier et résoudre les problèmes

## Compatibilité

Ces classes sont compatibles avec :
- Java 17+
- TensorFlow 0.5.0
- OpenCV 4.7.0-1.5.9
- Spring Boot 3.x