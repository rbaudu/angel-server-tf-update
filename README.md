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

## Modifications apportées par rapport à la version 1.0.0-rc.2

1. **Création des tenseurs TensorFlow**
   - Utilisation de la méthode factory `TUint8.tensorOf(shape, consumer)` pour les images non normalisées
   - Utilisation de la méthode factory `TFloat32.tensorOf(shape, consumer)` pour les images normalisées
   - Emploi de `StdArrays.copyTo()` pour copier les données dans les tenseurs

2. **Accès aux signatures du modèle**
   - Utilisation de `Map<String, Signature> signatures = model.signatures()` au lieu de `model.signatures().forEach((signature, signatureInfo) -> ...)`
   - Exploration correcte des entrées et sorties des signatures

3. **Extraction des données des tenseurs**
   - Utilisation de `resultTensor.data().get(resultArray)` pour obtenir les données du tenseur
   - Vérification des dimensions du tenseur avec `tensor.size()`

## Comment utiliser ces mises à jour

1. **Prérequis**
   - Java 17 ou supérieur
   - Maven pour la gestion des dépendances
   - Spring Boot 3.x

2. **Mise à jour du pom.xml**
   Assurez-vous que votre fichier pom.xml inclut la dépendance TensorFlow 0.5.0 :
   ```xml
   <dependency>
       <groupId>org.tensorflow</groupId>
       <artifactId>tensorflow-core-platform</artifactId>
       <version>0.5.0</version>
   </dependency>
   ```
   
   Vous pouvez supprimer les anciennes dépendances TensorFlow si présentes :
   ```xml
   <!-- À supprimer -->
   <dependency>
       <groupId>org.tensorflow</groupId>
       <artifactId>tensorflow-core-api</artifactId>
       <version>${tensorflow.version}</version>
   </dependency>
   <dependency>
       <groupId>org.tensorflow</groupId>
       <artifactId>tensorflow-core-native</artifactId>
       <version>${tensorflow.version}</version>
       <classifier>windows-x86_64</classifier>
   </dependency>
   ```

3. **Intégration des classes mises à jour**
   - Copiez les fichiers Java depuis ce dépôt vers votre projet
   - Respectez l'organisation des packages et des noms de classes

4. **Configuration des logs pour le débogage**
   Ajoutez cette configuration dans votre fichier `application.properties` :
   ```properties
   # Niveaux de logs pour le débogage TensorFlow
   logging.level.com.rbaudu.angel.analyzer.util=DEBUG
   logging.level.com.rbaudu.angel.analyzer.service.video=DEBUG
   logging.level.org.tensorflow=INFO
   ```

## Fonctionnalités de diagnostic

La nouvelle classe `TensorFlowDiagnostics` peut être utilisée pour diagnostiquer les problèmes liés à TensorFlow :

```java
@Service
public class ModelDiagnosticsService {
    private final TensorFlowDiagnostics diagnostics;
    private final ModelLoader modelLoader;
    
    @Autowired
    public ModelDiagnosticsService(TensorFlowDiagnostics diagnostics, ModelLoader modelLoader) {
        this.diagnostics = diagnostics;
        this.modelLoader = modelLoader;
    }
    
    public void analyzeModel(String modelPath) throws Exception {
        SavedModelBundle model = modelLoader.loadModel(modelPath);
        
        // Lister toutes les opérations du modèle
        diagnostics.listOperations(model);
        
        // Analyser les signatures du modèle
        diagnostics.analyzeSignatures(model);
        
        // Vérifier si le modèle attend des entrées normalisées
        boolean needsNormalization = diagnostics.doesModelExpectNormalizedInputs(model);
        System.out.println("Le modèle nécessite des entrées normalisées : " + needsNormalization);
    }
}
```

## Compatibilité des modèles

Ces classes ont été testées avec :
- MobileNetV2 pour la classification d'activités
- Modèles de détection d'objets basés sur SSD/Faster R-CNN

Pour d'autres types de modèles, vous pourriez avoir besoin d'ajuster légèrement les noms des tenseurs d'entrée/sortie.

## Résolution des problèmes

### 1. Erreurs de forme de tenseur
Si vous rencontrez des erreurs liées à la forme des tenseurs, utilisez la méthode `videoUtils.debugTensor()` pour vérifier les dimensions.

### 2. Erreurs de type de données
Assurez-vous d'utiliser le bon format (uint8 vs float32) en fonction des attentes de votre modèle. Utilisez la méthode `TensorFlowDiagnostics.doesModelExpectNormalizedInputs()` pour détecter automatiquement ce besoin.

### 3. Problèmes de performance
Si les performances sont insuffisantes, essayez d'utiliser des tenseurs uint8 plutôt que float32 lorsque possible, car la conversion est plus rapide.

## Contribution

N'hésitez pas à contribuer à ce projet en soumettant des pull requests ou en signalant des problèmes.

## Licence

Ce projet est distribué sous licence MIT, voir le fichier LICENSE pour plus de détails.
