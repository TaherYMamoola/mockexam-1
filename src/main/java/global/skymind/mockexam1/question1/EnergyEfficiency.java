package global.skymind.mockexam1.question1;

/*
 * Before you start, you should:
 * 1. Put the CSV file into your resources folder

 * You are an engineer and have collected a dataset on how different shape of buildings affect the energy efficiency.
 * You have to perform an analysis to determine:
 * 1. the heating load and;
 * 2. cooling load of the building,
 * given a set of attributes which correspond to a shape. Here, the hints of what you should do is given below:

 * Hint for transform process:
 * a) remove unused columns
 * b) filter out empty rows

 * Hint for model training: Perform early stopping so that the model does not overfit
 * Specifications:-
 * a) Calculate average loss on dataset
 * b) terminate if there is no improvement for 1 epoch
 * c) check the loss for each epoch

 * Dataset origin: https://archive.ics.uci.edu/ml/datasets/Energy+efficiency#
 * Dataset attribute description:
 * X1: Relative Compactnessgit s
 * X2: Surface Area
 * X3: Wall Area
 * X4: Roof Area
 * X5: Overall Height
 * X6: Orientation
 * X7: Glazing Area
 * X8: Glazing Area Distribution
 * Y1: Heating Load
 * Y2: Cooling Load
 */

import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class EnergyEfficiency {

    //===============Tunable parameters============
    private static int batchSize = 50;
    private static int seed = 123;
    private static double trainFraction = 0.8;
    private static double lr = 0.001;
    //=============================================
    private static TransformProcess tp;
    private static DataSet trainSet;
    private static DataSet testSet;
    private static RegressionEvaluation evalTrain;
    private static RegressionEvaluation evalTest;

    public static void main(String[] args) throws IOException, InterruptedException {

        File filePath = new ClassPathResource("EnergyEfficiency/ENB2012_data.csv").getFile();
        /*
         * Enter your code here
         */

        Schema schema = new Schema.Builder()
                .addColumnsDouble("rel-compactness","surface-area","wall-area","roof-area","overall-height")
                .addColumnInteger("orientation")
                .addColumnDouble("glazing-area")
                .addColumnInteger("glazing-area-distribution")
                .addColumnsDouble("heating-load","cooling-load")
                .addColumnsString("emptyCol1","emptyCol2")
                .build();

        /*
         * Enter your code here
         */

        List<List<Writable>> data = new ArrayList<>();

        /*
         * Enter your code here
         */

        List<List<Writable>> transformed = LocalTransformExecutor.execute(data, tp);
        System.out.println("=======Initial Schema=========\n"+ tp.getInitialSchema());
        System.out.println("=======Final Schema=========\n"+ tp.getFinalSchema());

        /*
         * Check the size of array before and after transformation
         * Please ensure that Pre-transformation size > Post-transformation size
         */
        System.out.println("Size before transform: "+ data.size()+
                "\nColumns before transform: "+ tp.getInitialSchema().numColumns());
        System.out.println("Size after transform: "+ transformed.size()+
                "\nColumns after transform: "+ tp.getFinalSchema().numColumns());

        CollectionRecordReader crr = new CollectionRecordReader(transformed);

        /*
         * Enter your code here
         */

        ViewIterator trainIter = new ViewIterator(trainSet, batchSize);
        ViewIterator testIter = new ViewIterator(testSet, batchSize);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .updater(new Adam(lr))
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(trainIter.inputColumns())
                        .nOut(50)
                        .nOut(50)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(2)
                        .activation(Activation.IDENTITY)
                        .build())
                .build();

        /*
         * Enter your code here
         */

        System.out.println(evalTrain.stats());
        System.out.println(evalTest.stats());

    }

}