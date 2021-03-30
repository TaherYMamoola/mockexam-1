package global.skymind.mockexam1.question2;

/*
  This dataset contains 10,000 images of t-shirt/top (class 0), trouser (class 1), pullover (class 2),
  dress (class 3), coat (class 4), sandal (class 5), shirt (class 6), sneaker (class 7), bag (class 8)
  and ankle boot (class 9).
  Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total.
  Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel,
  with higher numbers meaning darker. This pixel-value is an integer between 0 and 255.
  In total, there are 785 columns. The first column consists of the class labels (see above),
  and represents the article of clothing. The rest of the columns contain the pixel-values of the associated image.
  Dataset link: https://www.kaggle.com/zalando-research/fashionmnist
  What you need to do:
  1. Create a DataSetIterator to prepare for shuffling and dataset split.
  2. Shuffle and split the dataset using the fraction given.
  3. Perform normalisation on the dataset.
  4. Write your model configuration for this fashion image classifier.
  5. Define the early stopping configuration using the following criteria:
     (a) use F1 score to as a basis of score calculation
     (b) set the early stopping trainer to stop if the score does not improve after 1 epoch
  6. Perform early stopping training and return the final model
  ====================================================================
  Total marks: 30
  Assessment will be based on:
  1. Code is executable without error
  2. Correct configuration details as per specifications (if any)
  3. Convergence of the network
  ====================================================================
  NOTE: Location of each step is marked using the following annotation:
     /*
     * Your code here
     */

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ViewIterator;


import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class FashionMnistClassifier {

    private static int seed = 1234;
    private static double trainFrac = 0.8;
    private static double lr = 0.001;
    private static int height = 28;
    private static int width = 28;
    private static int channel = 1;
    private static int batchSize = 100;

    public static void main(String[] args) throws IOException, InterruptedException {

        CSVRecordReader csvrr = new CSVRecordReader(1, ',');
        csvrr.initialize(new FileSplit(getFile()));

        List<List<Writable>> originalData = new ArrayList<>();
        /*
         * Your code here
         */
        System.out.println("Size before transform: "+ originalData.size());

        List<List<Writable>> trainData = new ArrayList<>();
        List<List<Writable>> testData = new ArrayList<>();

        int numTrainData = (int) Math.round(trainFrac * originalData.size());
        int idx = 0;

        while (csvrr.hasNext()) {
            if (idx < numTrainData) {
                trainData.add(csvrr.next());
            } else {
                testData.add(csvrr.next());
            }
            idx++;
        }

        Schema schema = getSchema();
        List<List<Writable>> transformedTrainData = getTransformedData(schema, trainData);
        List<List<Writable>> transformedTestData = getTransformedData(schema, testData);
        System.out.println("Train data size: " + transformedTrainData.size());
        System.out.println("Test data size: " + transformedTestData.size());

        /*
         * Your code here
         */

        /*
         * Your code here
         */

        ViewIterator trainIter = new ViewIterator(trainSet, batchSize);
        ViewIterator testIter = new ViewIterator(testSet, batchSize);

        /*
         * Your code here
         */

        /*
         * Your code here
         */

        /*
         * Your code here
         */
        System.out.println(model.summary());

        Evaluation trainEval = model.evaluate(trainIter);
        Evaluation testEval = model.evaluate(testIter);

        System.out.println("Train Evaluation: \n" + trainEval.stats());
        System.out.println("Test Evaluation: \n" + testEval.stats());

    }

    private static File getFile() throws IOException {

        return new ClassPathResource("FashionMnist/fashion-mnist.csv").getFile();
    }

    private static Schema getSchema() {

        return new Schema.Builder()
                .addColumnInteger("label")
                .addColumnsInteger("pixel%d", 1, 784)
                .build();
    }

    private static List<List<Writable>> getTransformedData(Schema schema, List<List<Writable>> originalData) {

        TransformProcess tp = new TransformProcess.Builder(schema)
                .filter(new FilterInvalidValues())
                .build();

        return LocalTransformExecutor.execute(originalData, tp);
    }

    private static DataSet makeDataSet(List<List<Writable>> data) {

        /*
         * Your code here
         */
    }

}
