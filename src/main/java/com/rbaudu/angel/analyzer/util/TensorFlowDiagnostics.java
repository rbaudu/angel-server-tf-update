package com.rbaudu.angel.analyzer.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.springframework.stereotype.Component;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * Classe utilitaire pour diagnostiquer les problèmes liés à TensorFlow
 */
@Component
public class TensorFlowDiagnostics {
    private static final Logger logger = LoggerFactory.getLogger(TensorFlowDiagnostics.class);

    /**
     * Affiche toutes les opérations disponibles dans un modèle TensorFlow
     * @param model Modèle à analyser
     */
    public void listOperations(SavedModelBundle model) {
        if (model == null) {
            logger.error("Impossible de lister les opérations : modèle null");
            return;
        }
        
        Graph graph = model.graph();
        List<String> operations = new ArrayList<>();
        
        graph.operations().forEachRemaining(op -> {
            operations.add(op.name() + " (type: " + op.type() + ")");
            
            // Afficher les entrées de l'opération
            int numInputs = op.numInputs();
            if (numInputs > 0) {
                logger.debug("  Entrées de {}: ", op.name());
                for (int i = 0; i < numInputs; i++) {
                    logger.debug("    {}", op.input(i));
                }
            }
            
            // Afficher les attributs de l'opération (si disponibles)
            try {
                if (op.attributes() != null) {
                    op.attributes().forEach((name, value) -> {
                        logger.debug("  Attribut {}: {}", name, value);
                    });
                }
            } catch (Exception e) {
                // Ignorer les erreurs potentielles lors de l'accès aux attributs
            }
        });
        
        logger.info("Opérations du modèle ({} opérations): {}", operations.size(), operations);
    }
    
    /**
     * Analyse un Tensor pour déterminer son type et sa forme
     * @param tensor Tensor à analyser
     * @param name Nom du tensor pour l'identification
     */
    public void analyzeTensor(Tensor tensor, String name) {
        if (tensor == null) {
            logger.error("Tensor {} est null", name);
            return;
        }
        
        try {
            logger.info("Tensor {}: type={}, shape={}",
                name,
                tensor.dataType(),
                formatShape(tensor.shape().asArray()));
            
            // Essayer d'obtenir des informations supplémentaires sur le tenseur
            logger.info("Tensor {} nombre total d'éléments: {}",
                name, tensor.numElements());
                
        } catch (Exception e) {
            logger.error("Erreur lors de l'analyse du tensor {}: {}", name, e.getMessage());
        }
    }
    
    /**
     * Formate un tableau de dimensions pour l'affichage
     */
    private String formatShape(long[] shape) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < shape.length; i++) {
            sb.append(shape[i]);
            if (i < shape.length - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }
    
    /**
     * Vérifie si un modèle attend des entrées normalisées (0-1) ou non normalisées (0-255)
     * Cette méthode tente de deviner en analysant les noms des opérations
     * @param model Modèle à analyser
     * @return true si le modèle semble attendre des entrées normalisées
     */
    public boolean doesModelExpectNormalizedInputs(SavedModelBundle model) {
        if (model == null) {
            logger.error("Impossible d'analyser les attentes de normalisation : modèle null");
            return false;
        }
        
        // Liste des noms d'opérations qui suggèrent une normalisation intégrée
        List<String> normalizationOps = List.of(
            "normalization", "normalize", "preprocessing", "divide", "div", "scale"
        );
        
        // Chercher ces opérations près du début du graphe
        List<Operation> operations = new ArrayList<>();
        model.graph().operations().forEachRemaining(operations::add);
        
        // Parcourir les 20 premières opérations (généralement le prétraitement est au début)
        for (int i = 0; i < Math.min(20, operations.size()); i++) {
            Operation op = operations.get(i);
            String opName = op.name().toLowerCase();
            String opType = op.type().toLowerCase();
            
            // Vérifier si l'opération semble être liée à la normalisation
            for (String normOp : normalizationOps) {
                if (opName.contains(normOp) || opType.contains(normOp)) {
                    logger.info("Le modèle semble contenir une normalisation intégrée (opération: {})", op.name());
                    return true;
                }
            }
            
            // Vérifier les attributs pour les constantes de division qui pourraient être 255.0
            try {
                if (op.attributes() != null && opType.contains("const")) {
                    op.attributes().forEach((name, value) -> {
                        if (value != null && value.toString().contains("255")) {
                            logger.info("Le modèle semble contenir une normalisation intégrée (constante 255 trouvée)");
                        }
                    });
                }
            } catch (Exception e) {
                // Ignorer les erreurs potentielles lors de l'accès aux attributs
            }
        }
        
        logger.info("Le modèle ne semble pas contenir de normalisation intégrée");
        return false;
    }
}