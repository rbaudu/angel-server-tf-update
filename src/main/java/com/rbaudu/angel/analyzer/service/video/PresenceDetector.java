package com.rbaudu.angel.analyzer.service.video;

import com.rbaudu.angel.analyzer.config.AnalyzerConfig;
import com.rbaudu.angel.analyzer.util.ModelLoader;
import com.rbaudu.angel.analyzer.util.VideoUtils;
import static org.bytedeco.opencv.global.opencv_core.*;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_objdetect.HOGDescriptor;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.types.TFloat32;
import org.springframework.beans.factory.annotation.Autowired;

import jakarta.annotation.PostConstruct;
import java.util.Arrays;
import java.util.List;

/**
 * Service responsable de la détection de présence humaine dans les images vidéo.
 */
@Service
public class PresenceDetector {
    private static final Logger logger = LoggerFactory.getLogger(PresenceDetector.class);
    
    private final ModelLoader modelLoader;
    private final VideoUtils videoUtils;
    private final AnalyzerConfig config;
    
    private SavedModelBundle model;
    private List<String> personClasses = Arrays.asList("person");
    
    /**
     * Constructeur avec injection de dépendances.
     */
    @Autowired
    public PresenceDetector(ModelLoader modelLoader, VideoUtils videoUtils, AnalyzerConfig config) {
        this.modelLoader = modelLoader;
        this.videoUtils = videoUtils;
        this.config = config;
    }
    
    /**
     * Initialisation du modèle après construction du bean.
     */
    @PostConstruct
    public void init() {
        try {
            // Afficher la version de TensorFlow pour débogage
            logger.info("Version de TensorFlow: {}", TensorFlow.version());
            
            String modelPath = config.getHumanDetectionModel();
            if (modelPath != null && !modelPath.isEmpty()) {
                logger.info("Chargement du modèle de détection de présence humaine: {}", modelPath);
                this.model = modelLoader.loadModel(modelPath);
                logger.info("Modèle de détection de présence humaine chargé avec succès");
                
                // Lister les signatures disponibles pour le débogage
                model.signatures().forEach((signature, signatureInfo) -> {
                    logger.info("Signature disponible: {} - {}", signature, signatureInfo);
                });
            } else {
                logger.warn("Aucun modèle de détection de présence humaine configuré");
            }
        } catch (Exception e) {
            logger.error("Erreur lors du chargement du modèle de détection de présence humaine", e);
        }
    }
    
    /**
     * Détecte si une personne est présente dans l'image.
     * @param frame Image à analyser
     * @return true si une personne est détectée, false sinon
     */
    public boolean isPersonPresent(Mat frame) {
        if (model == null) {
            logger.warn("Détection de présence impossible : modèle non chargé");
            return false;
        }
        
        try {
            // Redimensionner et prétraiter l'image
            Mat processedFrame = videoUtils.resizeFrame(frame, 320, 320);
            Tensor imageTensor = videoUtils.prepareImageForModel(processedFrame, 320, 320, false);
            
            // Déboguer le tensor d'entrée
            videoUtils.debugTensor(imageTensor, "Tensor d'entrée pour la détection de présence");
            
            // Exécuter l'inférence avec la nouvelle API TensorFlow
            Session.Runner runner = model.session().runner()
                    .feed("serving_default_input_tensor", imageTensor)
                    .fetch("StatefulPartitionedCall:1")  // Classes
                    .fetch("StatefulPartitionedCall:2"); // Scores
            
            List<Tensor> outputs = runner.run();
            TFloat32 resultClassTensor = (TFloat32) outputs.get(0);
            TFloat32 resultScoreTensor = (TFloat32) outputs.get(1);
            
            // Accéder aux données des tensors avec la nouvelle API
            float[] resultClassArray = new float[(int)resultClassTensor.size()];
            float[] resultScoreArray = new float[(int)resultScoreTensor.size()];
            
            // Copier les données des tensors dans les tableaux
            resultClassTensor.data().get(resultClassArray);
            resultScoreTensor.data().get(resultScoreArray);
            
            // Chercher les détections de personnes
            for (int i = 0; i < Math.min(resultClassArray.length, resultScoreArray.length); i++) {
                logger.debug("Détection avec un score de {} et une classe de {}", 
                        resultScoreArray[i], resultClassArray[i]);

                if (resultScoreArray[i] > config.getPresenceThreshold()) {
                    int classId = (int) resultClassArray[i];  // Convertir en entier
                    if (classId == 1) {  // Classe 1 = personne
                        logger.debug("Personne détectée avec un score de {}", resultScoreArray[i]);
                        return true;
                    }
                }
            }
            
            logger.debug("Aucune personne détectée");
            return false;
            
        } catch (Exception e) {
            logger.error("Erreur lors de la détection de présence", e);
            e.printStackTrace();
            return false;
        }
    }
    
    /**
     * Alternative basée sur OpenCV pour la détection de personnes.
     * Utilisé comme solution de secours si TensorFlow ne fonctionne pas.
     */
    public boolean detectPersonWithHOG(Mat frame) {
        try {
            HOGDescriptor hog = new HOGDescriptor();
            
            // Dans OpenCV 4.x, on peut utiliser le détecteur de personnes par défaut directement
            FloatPointer detector = HOGDescriptor.getDefaultPeopleDetector();
            Mat detectorMat = new Mat(detector.limit(), 1, CV_32F); 
            // Convertir BytePointer en FloatPointer et copier les données
            FloatPointer detectorMatPointer = new FloatPointer(detectorMat.data());

            hog.setSVMDetector(detectorMat);
            
            // Conteneurs pour les résultats
            RectVector foundLocations = new RectVector();
            DoublePointer weights = new DoublePointer();
            
            // Redimensionner pour de meilleures performances
            Mat resizedFrame = videoUtils.resizeFrame(frame, 640, 480);
            
            // Détecter les personnes
            hog.detectMultiScale(resizedFrame, foundLocations, weights);
            
            return foundLocations.size() > 0;
        } catch (Exception e) {
            logger.error("Erreur lors de la détection de personne avec HOG", e);
            return false;
        }
    }
}