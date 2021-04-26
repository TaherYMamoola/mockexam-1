package global.skymind.mockexam1.question3;

//import global.skymind.mockexam1.HorseBreedIterator;

import org.datavec.image.transform.*;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
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

        ImageTransform hflip = new FlipImageTransform(0);
        ImageTransform crop = new CropImageTransform(5);
        ImageTransform rotation = new RotateImageTransform(15);
        ImageTransform randCrop = new RandomCropTransform(50,50);

        List<Pair<ImageTransform,Double>> pipeline = Arrays.asList(
                new Pair<>(hflip,0.2),
                new Pair<>(crop,0.2),
                new Pair<>(rotation,0.2),
                new Pair<>(randCrop,0.2)
        );

        ImageTransform transform = new PipelineImageTransform(pipeline,false);

        DataSetIterator trainIter = HorseBreedIterator.getTrain(transform,batchSize);
        DataSetIterator testIter = HorseBreedIterator.getTest(1);

        // Build your model configuration

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .updater(new Adam(lr))
                .weightInit(WeightInit.XAVIER)
                .seed(seed)
                .l2(0.001)
                .list()
                .layer(0,new ConvolutionLayer.Builder()
                        .kernelSize(3,3)
                        .stride(1,1)
                        .activation(Activation.RELU)
                        .nIn(nChannel)
                        .nOut(500)
                        .build())
                .layer(1,new SubsamplingLayer.Builder()
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2,new ConvolutionLayer.Builder()
                        .nOut(250)
                        .kernelSize(3,3)
                        .stride(1,1)
                        .activation(Activation.RELU)
                        .build())
                .layer(3,new SubsamplingLayer.Builder()
                        .stride(2,2)
                        .kernelSize(2,2)
                        .poolingType(PoolingType.MAX)
                        .build())
                .layer(4,new ConvolutionLayer.Builder()
                        .nOut(100)
                        .kernelSize(3,3)
                        .stride(1,1)
                        .activation(Activation.RELU)
                        .build())
                .layer(5,new SubsamplingLayer.Builder()
                        .stride(2,2)
                        .kernelSize(2,2)
                        .poolingType(PoolingType.AVG)
                        .build())
                .layer(6,new ConvolutionLayer.Builder()
                        .nOut(50)
                        .kernelSize(3,3)
                        .stride(1,1)
                        .activation(Activation.RELU)
                        .build())
                .layer(7,new SubsamplingLayer.Builder()
                        .stride(2,2)
                        .kernelSize(2,2)
                        .poolingType(PoolingType.MAX)
                        .build())
                .layer(8,new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(25)
                        .build())
                .layer(9,new OutputLayer.Builder()
                        .nOut(nOutput)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .setInputType(InputType.convolutional(height,width,nChannel))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        log.info("**************************************** MODEL SUMMARY ****************************************");
        System.out.println(model.summary());

        // Train your model and set listeners

        model.setListeners(new ScoreIterationListener(10));
        model.fit(trainIter,nEpoch);

        log.info("**************************************** MODEL EVALUATION ****************************************");

        // Perform evaluation on both train and test set

        Evaluation trainEval = model.evaluate(trainIter);
        Evaluation testEval = model.evaluate(testIter);
        System.out.println("Train Eval : "+ trainEval.stats());
        System.out.println("Test Eval : "+ testEval.stats());

    }

}
