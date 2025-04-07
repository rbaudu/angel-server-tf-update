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
   - Utilisation de `List<Signature> signatures = model.signatures()` au lieu de `Map<String, Signature>`
   - Exploration correcte des entrées et sorties des signatures

3. **Extraction des données des tenseurs**
   - Utilisation de `FloatBuffer` et `copyTo()` pour extraire les données du tenseur
   - Par exemple : 
     ```java
     FloatBuffer buffer = FloatBuffer.allocate(size);
     resultTensor.copyTo(buffer);
     buffer.rewind();
     buffer.get(resultArray);
     ```

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

## Différences API importantes

### 1. Conversion des images (VideoUtils)

**Avant** (TensorFlow 1.0.0-rc.2) :
```java
// Méthode problématique
float[] pixelData = new float[height * width * channels];
// Remplissage manuel des données...
return Tensor.create(new long[]{1, height, width, 3}, floatBuffer);
```

**Après** (TensorFlow 0.5.0) :
```java
// Approche moderne avec des tableaux multidimensionnels
float[][][][] pixelData = new float[1][height][width][channels];
// Remplissage...
return TFloat32.tensorOf(shape, data -> StdArrays.copyTo(pixelData, data));
```

### 2. Accès aux signatures du modèle

**Avant** :
```java
model.signatures().forEach((signature, signatureInfo) -> {
    // Traitement des signatures...
});
```

**Après** :
```java
List<Signature> signatures = model.signatures();
signatures.forEach(signature -> {
    logger.info("Signature: {}", signature.key());
    // Accès aux entrées/sorties...
});
```

### 3. Extraction des données d'un tensor

**Avant** :
```java
float[][][] result = new float[1][100][7];
resultTensor.copyTo(result);
```

**Après** :
```java
float[] resultArray = new float[(int)resultTensor.size()];
FloatBuffer buffer = FloatBuffer.allocate(resultArray.length);
resultTensor.copyTo(buffer);
buffer.rewind();
buffer.get(resultArray);
```

## Guide de dépannage

### Erreurs courantes

1. **NoSuchMethodError** : Si vous rencontrez une erreur `NoSuchMethodError`, c'est probablement parce que la méthode n'existe pas dans TensorFlow 0.5.0 ou a une signature différente.

2. **ClassCastException** : Vérifiez que vous utilisez les bons types (TFloat32, TUint8) pour vos tenseurs.

3. **Shape mismatch** : Si le modèle attend une forme différente de celle que vous fournissez, utilisez la classe `TensorFlowDiagnostics` pour analyser les signatures et comprendre ce que le modèle attend.

### Solutions

1. **Utilisez TensorFlowDiagnostics** : Cette classe vous aidera à identifier les problèmes avec vos modèles et tenseurs.

```java
// Exemple d'utilisation de TensorFlowDiagnostics
@Autowired
private TensorFlowDiagnostics diagnostics;

// Dans votre méthode
SavedModelBundle model = modelLoader.loadModel(modelPath);
diagnostics.analyzeSignatures(model);
```

2. **Vérifiez les logs** : Augmentez le niveau de logs pour TensorFlow et votre application pour voir les détails des erreurs.

3. **Testez avec des images simples** : Commencez par tester avec des images simples et des tailles réduites pour vérifier que le processus fonctionne.

## Compatibilité des modèles

Les modèles TensorFlow SavedModel existants devraient continuer à fonctionner avec cette mise à jour. Si vous rencontrez des problèmes, vérifiez :

1. Les noms des tenseurs d'entrée et de sortie (ils doivent correspondre exactement)
2. Les formes des tenseurs d'entrée et de sortie
3. Le besoin ou non de normalisation des entrées

## Support et contribution

Si vous rencontrez des problèmes avec ces mises à jour, veuillez ouvrir une issue dans ce dépôt ou contribuer avec vos propres améliorations via une pull request.

Pour plus d'informations sur l'API TensorFlow Java 0.5.0, consultez la [documentation officielle](https://www.tensorflow.org/jvm).
