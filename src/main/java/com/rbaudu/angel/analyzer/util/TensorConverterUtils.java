package com.rbaudu.angel.analyzer.util;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.javacpp.BytePointer;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TUint8;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.Arrays;

/**
 * Utilitaires spécialisés pour la conversion entre différents formats de tenseurs.
 * Cette classe fournit des méthodes pour manipuler et convertir des tenseurs
 * entre différents types de données, formats et frameworks.
 */
@Component
public class TensorConverterUtils {
    private static final Logger logger = LoggerFactory.getLogger(TensorConverterUtils.class);
    
    /**
     * Convertit un tensor TensorFlow Uint8 en tensor Float32 en effectuant une normalisation.
     * Cette méthode est utile pour adapter des modèles qui attendent des entrées normalisées
     * mais qui sont alimentés avec des données brutes.
     *
     * @param uint8Tensor Le tensor source au format uint8
     * @return Un nouveau tensor TFloat32 contenant les mêmes données mais normalisées entre 0 et 1
     */
    public Tensor convertUint8ToFloat32(Tensor uint8Tensor) {
        if (!(uint8Tensor instanceof TUint8)) {
            throw new IllegalArgumentException("Le tensor d'entrée doit être de type TUint8");
        }
        
        TUint8 input = (TUint8) uint8Tensor;
        Shape shape = input.shape();
        
        // Obtenir les dimensions
        long[] dims = shape.asArray();
        int totalSize = 1;
        for (long dim : dims) {
            totalSize *= dim;
        }
        
        // Créer un array pour stocker les données normalisées
        byte[] rawData = new byte[totalSize];
        input.data().get(rawData);
        
        // Normaliser les données
        float[] normalizedData = new float[totalSize];
        for (int i = 0; i < totalSize; i++) {
            normalizedData[i] = (rawData[i] & 0xFF) / 255.0f;
        }
        
        // Créer le nouveau tensor TFloat32
        TFloat32 output = TFloat32.tensorOf(shape);
        
        // Déterminer les dimensions pour StdArrays.ndCopyOf
        if (dims.length == 4) { // [batch, height, width, channels]
            float[][][][] reshapedData = reshapeToNDArray(normalizedData, (int)dims[0], (int)dims[1], (int)dims[2], (int)dims[3]);
            output.setData(StdArrays.ndCopyOf(reshapedData));
        } else {
            logger.warn("Conversion de forme de tensor non supportée: {}. Utilisation d'une méthode alternative.", Arrays.toString(dims));
            // Méthode alternative si la forme n'est pas 4D
            output.setData(StdArrays.of(normalizedData).reshape(dims));
        }
        
        return output;
    }
    
    /**
     * Convertit un tableau 1D en tableau 4D
     * @param data Tableau plat 1D
     * @param batch Taille du batch
     * @param height Hauteur
     * @param width Largeur
     * @param channels Nombre de canaux
     * @return Tableau 4D redimensionné
     */
    private float[][][][] reshapeToNDArray(float[] data, int batch, int height, int width, int channels) {
        float[][][][] result = new float[batch][height][width][channels];
        int index = 0;
        
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    for (int c = 0; c < channels; c++) {
                        result[b][h][w][c] = data[index++];
                    }
                }
            }
        }
        
        return result;
    }
    
    /**
     * Convertit un tensor TensorFlow Float32 en image OpenCV
     * @param tensor Le tensor source au format float32
     * @return Une image OpenCV (Mat)
     */
    public Mat tensorToMat(Tensor tensor) {
        if (!(tensor instanceof TFloat32) && !(tensor instanceof TUint8)) {
            throw new IllegalArgumentException("Le tensor doit être de type TFloat32 ou TUint8");
        }
        
        Shape shape = tensor.shape();
        long[] dims = shape.asArray();
        
        if (dims.length != 4 || dims[0] != 1) {
            throw new IllegalArgumentException("Le tensor doit avoir une forme [1, height, width, channels]");
        }
        
        int height = (int) dims[1];
        int width = (int) dims[2];
        int channels = (int) dims[3];
        
        // Créer une nouvelle image OpenCV
        Mat image = new Mat(height, width, channels == 3 ? 16 : 8); // CV_8UC3 ou CV_8UC1
        
        // Copier les données du tensor vers l'image
        if (tensor instanceof TFloat32) {
            // Obtenir les données normalisées
            float[] data = new float[(int) tensor.size()];
            ((TFloat32) tensor).data().get(data);
            
            // Convertir en bytes (dénormaliser si nécessaire)
            byte[] imageData = new byte[height * width * channels];
            for (int i = 0; i < data.length; i++) {
                imageData[i] = (byte) (data[i] * 255.0f);
            }
            
            // Copier dans l'image OpenCV
            BytePointer imagePtr = image.data();
            imagePtr.put(imageData);
        } else {
            // C'est un TUint8, copier directement
            byte[] data = new byte[(int) tensor.size()];
            ((TUint8) tensor).data().get(data);
            
            // Copier dans l'image OpenCV
            BytePointer imagePtr = image.data();
            imagePtr.put(data);
        }
        
        return image;
    }
}