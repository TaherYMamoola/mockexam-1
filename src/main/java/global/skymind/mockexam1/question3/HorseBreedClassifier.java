package global.skymind.mockexam1.question3;

import global.skymind.mockexam1.HorseBreedIterator;
import org.datavec.image.transform.*;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;

import java.io.IOException;
import java.util.Random;

/* ===================================================================
 * We will solve a task of classifying horse breeds.
 * The dataset contains 4 classes, each with just over 100 images
 * Images are of 256x256 RGB
 *
 * Source: https://www.kaggle.com/olgabelitskaya/horse-breeds
 * ===================================================================
 * TO-DO
 *
 * 1. In HorseBreedIterator complete both methods (i) setup and (ii) makeIterator
 * 2. Complete ImageTransform Pipeline
 * 3. Complete your network configuration
 * 4. Train your model and set listeners
 * 5. Perform evaluation on both train and test set
 * 6. [OPTIONAL] Mitigate the underfitting problem
 *
 * ====================================================================
 * Assessment will be based on
 *
 * 1. Correct and complete configuration details
 * 2. HorseBreedClassifier is executable
 * 3. Convergence of the network
 * 4. Accuracy for both train and test are both over 85%
 * ====================================================================
 ** NOTE: Only make changes at sections with the following. Replace accordingly.
 *
 *   /*
 *    *
 *    * WRITE YOUR CODES HERE
 *    *
 *    *
 */

public class HorseBreedClassifier {

    private static final Logger log = org.slf4j.LoggerFactory.getLogger(HorseBreedClassifier.class);
    private static final int height = 64;
    private static final int width = 64;
    private static final int nChannel = 3;
    private static final int nOutput = 4;
    private static final int seed = 141;
    private static Random rng = new Random(seed);
    private static double lr = 1e-4;
    private static final int nEpoch = 20;
    private static final int batchSize = 3;

    public static void main(String[] args) throws IOException {

        HorseBreedIterator.setup();

        // Build an Image Transform pipeline consisting of
        // a horizontal flip, crop, rotation, and random cropping

        /*
         *
         *
         *  Write your codes here
         *
         *
         */

        ImageTransform transform = new PipelineImageTransform();

        DataSetIterator trainIter = HorseBreedIterator.getTrain(transform,batchSize);
        DataSetIterator testIter = HorseBreedIterator.getTest(1);

        // Build your model configuration

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                /*
                 *
                 *
                 *  Write your codes here
                 *
                 *
                 */
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        log.info("**************************************** MODEL SUMMARY ****************************************");
        System.out.println(model.summary());

        // Train your model and set listeners

        /*
         *
         *
         *  Write your codes here
         *
         *
         */

        log.info("**************************************** MODEL EVALUATION ****************************************");

        // Perform evaluation on both train and test set

        /*
         *
         *
         *  Write your codes here
         *
         *
         */

    }

}
