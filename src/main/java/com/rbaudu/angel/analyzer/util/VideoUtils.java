package com.rbaudu.angel.analyzer.util;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.springframework.stereotype.Component;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TUint8;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;

/**
 * Utilitaire pour traiter les images vidéo.
 */
@Component
public class VideoUtils {
    private static final Logger logger = LoggerFactory.getLogger(VideoUtils.class);
    
    /**
     * Redimensionne une image pour l'entrée du modèle
     * @param frame Image à redimensionner
     * @param width Largeur cible
     * @param height Hauteur cible
     * @return Image redimensionnée
     */
    public Mat resizeFrame(Mat frame, int width, int height) {
        Mat resized = new Mat();
        Size size = new Size(width, height);
        opencv_imgproc.resize(frame, resized, size);
        return resized;
    }
    
    /**
     * Convertit une image BGR en RGB
     * @param frame Image BGR
     * @return Image RGB
     */
    public Mat bgrToRgb(Mat frame) {
        Mat rgb = new Mat();
        opencv_imgproc.cvtColor(frame, rgb, opencv_imgproc.COLOR_BGR2RGB);
        return rgb;
    }
    
    /**
     * Normalise les valeurs de pixels (0-255 vers 0-1)
     * @param frame Image à normaliser
     * @return Image normalisée
     */
    public Mat normalizeFrame(Mat frame) {
        Mat normalized = new Mat();
        frame.convertTo(normalized, frame.type(), 1.0/255, 0);
        return normalized;
    }
    
    /**
     * Convertit une image OpenCV en Tensor TensorFlow au format uint8 (0-255)
     * @param frame Image OpenCV (RGB)
     * @param height Hauteur du tensor
     * @param width Largeur du tensor
     * @return Tensor TensorFlow
     */
    public Tensor matToTensorUint8(Mat frame, int height, int width) {
        int channels = frame.channels();
        
        // Format: [1, height, width, 3]
        // Créer un tableau multidimensionnel pour stocker les données
        byte[][][][] pixelData = new byte[1][height][width][channels];
        
        // Conversion des données OpenCV en tableau de bytes
        BytePointer bytePtr = frame.ptr();
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int pos = y * width * channels + x * channels;
                for (int c = 0; c < channels; c++) {
                    // Conserver les valeurs brutes (0-255)
                    pixelData[0][y][x][c] = bytePtr.get(pos + c);
                }
            }
        }
        
        // Création du tensor uint8 avec la nouvelle API TensorFlow
        Shape shape = Shape.of(1, height, width, channels);
        TUint8 tensor = TUint8.tensorOf(shape);
        
        // Copier les données dans le tensor avec le tableau multidimensionnel
        tensor.setData(StdArrays.ndCopyOf(pixelData));
        
        logger.debug("Tensor uint8 créé avec succès, forme: {}", Arrays.toString(shape.asArray()));
        return tensor;
    }
    
    /**
     * Convertit une image OpenCV en Tensor TensorFlow au format float32 (0-1)
     * @param frame Image OpenCV (RGB)
     * @param height Hauteur du tensor
     * @param width Largeur du tensor
     * @return Tensor TensorFlow TFloat32
     */
    public Tensor matToTensorFloat32(Mat frame, int height, int width) {
        int channels = frame.channels();
        
        // Créer un tableau pour les pixels normalisés
        float[][][][] pixelData = new float[1][height][width][channels];
        
        try (UByteRawIndexer indexer = frame.createIndexer()) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    for (int c = 0; c < channels; c++) {
                        // Normalisation entre 0-1
                        pixelData[0][y][x][c] = indexer.get(y, x, c) / 255.0f;
                    }
                }
            }
        }
        
        // Création du tensor TFloat32 avec la nouvelle API TensorFlow
        Shape shape = Shape.of(1, height, width, channels);
        TFloat32 tensor = TFloat32.tensorOf(shape);
        
        // Copier les données dans le tensor
        tensor.setData(StdArrays.ndCopyOf(pixelData));
        
        logger.debug("Tensor float32 créé avec succès, forme: {}", Arrays.toString(shape.asArray()));
        return tensor;
    }
    
    /**
     * Prépare une image OpenCV pour l'entrée du modèle TensorFlow
     * @param frame Image source
     * @param targetWidth Largeur cible
     * @param targetHeight Hauteur cible
     * @param asFloat32 Si true, convertit en TFloat32 (0-1), sinon en TUint8 (0-255)
     * @return Tensor prêt pour l'inférence
     */
    public Tensor prepareImageForModel(Mat frame, int targetWidth, int targetHeight, boolean asFloat32) {
        // Redimensionner et convertir en RGB
        Mat resized = resizeFrame(frame, targetWidth, targetHeight);
        Mat rgb = bgrToRgb(resized);
        
        // Conversion en tensor selon le type demandé
        if (asFloat32) {
            // Pour les modèles qui attendent des valeurs normalisées (0-1)
            return matToTensorFloat32(rgb, targetHeight, targetWidth);
        } else {
            // Pour les modèles qui attendent des valeurs brutes (0-255)
            return matToTensorUint8(rgb, targetHeight, targetWidth);
        }
    }
    
    /**
     * Affiche les informations sur un Tensor pour le débogage
     * @param tensor Tensor à analyser
     * @param name Nom du tensor pour l'identification dans les logs
     */
    public void debugTensor(Tensor tensor, String name) {
        logger.debug("{} - Type: {}, Shape: {}", 
            name,
            tensor.getClass().getSimpleName(),
            Arrays.toString(tensor.shape().asArray()));
        
        // On pourrait ajouter du code pour échantillonner quelques valeurs du tensor
        // si nécessaire pour le débogage
    }
}