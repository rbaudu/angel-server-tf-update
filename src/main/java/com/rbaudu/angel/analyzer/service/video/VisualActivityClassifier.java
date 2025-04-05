package com.rbaudu.angel.analyzer.service.video;

import com.rbaudu.angel.analyzer.config.AnalyzerConfig;
import com.rbaudu.angel.analyzer.model.ActivityType;
import com.rbaudu.angel.analyzer.util.ModelLoader;
import com.rbaudu.angel.analyzer.util.VideoUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.types.TFloat32;

import jakarta.annotation.PostConstruct;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Service de classification d'activités basé sur l'analyse vidéo.
 */
@Service
public class VisualActivityClassifier {
    private static final Logger logger = LoggerFactory.getLogger(VisualActivityClassifier.class);
    
    private final ModelLoader modelLoader;
    private final VideoUtils videoUtils;
    private final AnalyzerConfig config;
    
    private SavedModelBundle model;
    
    /**
     * Constructeur avec injection de dépendances.
     * @param modelLoader Chargeur de modèle TensorFlow
     * @param videoUtils Utilitaires vidéo
     * @param config Configuration de l'analyseur
     */
    public VisualActivityClassifier(ModelLoader modelLoader, VideoUtils videoUtils, AnalyzerConfig config) {
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
            
            if (config.getActivityRecognitionModel() != null) {
                this.model = modelLoader.loadModel(config.getActivityRecognitionModel());
                logger.info("Modèle de classification d'activités chargé avec succès");
                
                // Lister les signatures disponibles pour le débogage
                model.signatures().forEach((signature, signatureInfo) -> {
                    logger.info("Signature disponible: {} - {}", signature, signatureInfo);
                });
            } else {
                logger.warn("Aucun modèle de classification d'activités configuré");
            }
        } catch (Exception e) {
            logger.error("Erreur lors du chargement du modèle de classification d'activités", e);
        }
    }
    
    /**
     * Classifie l'activité visible dans l'image.
     * @param frame Image à analyser
     * @return Map des types d'activités avec leur score de confiance
     */
    public Map<ActivityType, Double> classifyActivity(Mat frame) {
        if (model == null) {
            logger.warn("Classification d'activités impossible : modèle non chargé");
            return new HashMap<>();
        }
        
        try {
            // Prétraitement de l'image - utiliser float32 pour MobileNetV2
            Tensor imageTensor = videoUtils.prepareImageForModel(
                    frame, 
                    config.getInputImageWidth(),
                    config.getInputImageHeight(),
                    true);  // true pour avoir des valeurs normalisées (0-1)
            
            // Déboguer le tensor d'entrée
            videoUtils.debugTensor(imageTensor, "Tensor d'entrée pour la classification d'activités");
            
            // Exécution de la classification avec TensorFlow
            Session.Runner runner = model.session().runner()
                    .feed("serving_default_inputs", imageTensor)
                    .fetch("StatefulPartitionedCall");
            
            List<Tensor> outputs = runner.run();
            TFloat32 resultTensor = (TFloat32) outputs.get(0);
            
            // Extraire les résultats du Tensor
            int numActivities = ActivityType.values().length - 1; // -1 pour exclure ABSENT
            float[] results = new float[numActivities];
            
            // Accéder aux données du tensor avec la nouvelle API
            resultTensor.data().get(results);
            
            // Conversion des probabilités en map
            Map<ActivityType, Double> result = new HashMap<>();
            for (int i = 0; i < numActivities; i++) {
                ActivityType activity = mapIndexToActivityType(i);
                if (activity != ActivityType.ABSENT) { // On exclut ABSENT de la classification visuelle
                    double probability = results[i];
                    if (probability > config.getActivityConfidenceThreshold()) {
                        result.put(activity, probability);
                    }
                }
            }
            
            logger.debug("Activités classifiées: {}", result);
            return result;
            
        } catch (Exception e) {
            logger.error("Erreur lors de la classification d'activités", e);
            e.printStackTrace();
            return new HashMap<>();
        }
    }
    
    /**
     * Mappe l'index de sortie du modèle à un type d'activité.
     * @param index Index dans le vecteur de sortie du modèle
     * @return Type d'activité correspondant
     */
    private ActivityType mapIndexToActivityType(int index) {
        // Mapping entre l'index de sortie du modèle et les types d'activités
        // À définir selon l'ordre des classes dans le modèle
        switch (index) {
            case 0: return ActivityType.CLEANING;
            case 1: return ActivityType.CONVERSING;
            case 2: return ActivityType.COOKING;
            case 3: return ActivityType.DANCING;
            case 4: return ActivityType.EATING;
            case 5: return ActivityType.FEEDING;
            case 6: return ActivityType.GOING_TO_SLEEP;
            case 7: return ActivityType.IRONING;
            case 8: return ActivityType.KNITTING;
            case 9: return ActivityType.LISTENING_MUSIC;
            case 10: return ActivityType.MOVING;
            case 11: return ActivityType.NEEDING_HELP;
            case 12: return ActivityType.PHONING;
            case 13: return ActivityType.PLAYING;
            case 14: return ActivityType.PLAYING_MUSIC;
            case 15: return ActivityType.PUTTING_AWAY;
            case 16: return ActivityType.READING;
            case 17: return ActivityType.RECEIVING;
            case 18: return ActivityType.SINGING;
            case 19: return ActivityType.SLEEPING;
            case 21: return ActivityType.UNKNOWN;
            case 22: return ActivityType.USING_SCREEN;
            case 23: return ActivityType.WAITING;
            case 24: return ActivityType.WAKING_UP;
            case 25: return ActivityType.WASHING;
            case 26: return ActivityType.WATCHING_TV;
            default: return ActivityType.WRITING;
        }
    }
}