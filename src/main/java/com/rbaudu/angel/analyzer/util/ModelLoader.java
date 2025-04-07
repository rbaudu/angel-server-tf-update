package com.rbaudu.angel.analyzer.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.TensorFlow;

import java.nio.file.Paths;
import java.util.Map;
import org.tensorflow.Signature;

/**
 * Utilitaire pour charger les modèles TensorFlow.
 */
@Component
public class ModelLoader {
    private static final Logger logger = LoggerFactory.getLogger(ModelLoader.class);
    
    /**
     * Charge un modèle TensorFlow à partir d'un chemin de fichier.
     * @param modelPath Chemin vers le répertoire du modèle SavedModel
     * @return Bundle du modèle chargé
     * @throws Exception En cas d'erreur lors du chargement
     */
    public SavedModelBundle loadModel(String modelPath) throws Exception {
        try {
            logger.info("Chargement du modèle TensorFlow depuis {}", modelPath);
            
            // Afficher la version de TensorFlow pour débogage
            logger.info("Version de TensorFlow: {}", TensorFlow.version());
            
            // Charger le modèle avec la nouvelle API
            SavedModelBundle model = SavedModelBundle.load(modelPath, "serve");
            
            // Lister les signatures disponibles pour le débogage
            Map<String, Signature> signatures = model.signatures();
            signatures.forEach((signatureKey, signature) -> {
                logger.info("Signature disponible: {}", signatureKey);
                logger.info("  Méthode: {}", signature.methodName());
                logger.info("  Entrées: {}", signature.inputNames());
                logger.info("  Sorties: {}", signature.outputNames());
            });
            
            logger.info("Modèle chargé avec succès");
            return model;
        } catch (Exception e) {
            logger.error("Erreur lors du chargement du modèle: {}", e.getMessage(), e);
            throw e;
        }
    }
    
    /**
     * Vérifie si un modèle existe à l'emplacement spécifié.
     * @param modelPath Chemin vers le répertoire du modèle
     * @return true si le modèle existe, false sinon
     */
    public boolean modelExists(String modelPath) {
        return Paths.get(modelPath, "saved_model.pb").toFile().exists();
    }
}