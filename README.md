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
   - Utilisation de `NdArrays` pour extraire les données du tenseur
   - Par exemple : 
     ```java
     FloatNdArray ndArray = NdArrays.ofFloats(Shape.of(size));
     resultTensor.copyTo(ndArray);
     float value = ndArray.getFloat(i);
     ```

4. **Gestion des résultats d'inférence**
   - Utilisation de `Result result = runner.run()` au lieu de `List<Tensor>`
   - Accès aux tenseurs avec `result.get(0)`, `result.get(1)`, etc.

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
FloatNdArray ndArray = NdArrays.ofFloats(Shape.of(size));
resultTensor.copyTo(ndArray);
float value = ndArray.getFloat(i);
```

### 4. Exécution de l'inférence

**Avant** :
```java
List<Tensor> outputs = runner.run();
TFloat32 resultTensor = (TFloat32) outputs.get(0);
```

**Après** :
```java
Result result = runner.run();
TFloat32 resultTensor = (TFloat32) result.get(0);
```

## Erreurs courantes et solutions

### 1. Création des NdArrays

**Erreur** : `The method ofFloats(Shape) in the type NdArrays is not applicable for the arguments (int)`

**Solution** : Utilisez `Shape.of()` pour créer un objet Shape à partir d'un entier :
```java
// Incorrect
FloatNdArray ndArray = NdArrays.ofFloats(numActivities);

// Correct
FloatNdArray ndArray = NdArrays.ofFloats(Shape.of(numActivities));
```

### 2. Extraction des données d'un tensor

**Erreur** : `The method data() is undefined for the type TFloat32`

**Solution** : Utilisez `copyTo()` avec un NdArray :
```java
// Incorrect
resultTensor.data().get(resultArray);

// Correct
FloatNdArray ndArray = NdArrays.ofFloats(Shape.of(size));
resultTensor.copyTo(ndArray);
```

### 3. Types de retour des méthodes

**Erreur** : `Type mismatch: cannot convert from Result to List<Tensor>`

**Solution** : Adaptez votre code pour utiliser la classe `Result` au lieu de `List<Tensor>` :
```java
// Incorrect
List<Tensor> outputs = runner.run();

// Correct
Result result = runner.run();
```

### 4. Importations manquantes

Pour éviter les erreurs d'importation, assurez-vous d'inclure :
```java
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.FloatNdArray;
```

## Support et contribution

Si vous rencontrez des problèmes avec ces mises à jour, veuillez ouvrir une issue dans ce dépôt ou contribuer avec vos propres améliorations via une pull request.

Pour plus d'informations sur l'API TensorFlow Java 0.5.0, consultez la [documentation officielle](https://www.tensorflow.org/jvm).
