package com.github.question2;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;


import java.io.File;
import java.io.IOException;
import java.util.Random;

/* ===================================================================
 *
 * TO-DO
 *
 * 1. Complete setup method
 * 2. Complete makeIterator
 *
 * ====================================================================
 ** NOTE: Only make changes at sections with the following. Replace accordingly.
 *
 *   /*
 *    *
 *    * WRITE YOUR CODES HERE
 *    *
 *    *
 */

public class HorseBreedIterator {

    private static final String[] allowedExt = BaseImageLoader.ALLOWED_FORMATS;
    private static final int height = 64;
    private static final int width = 64;
    private static final int nChannel = 3;
    private static final int numLabels = 4;
    private static final int seed = 141;
    private static int batchSize;
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static Random rng = new Random(seed);
    private static InputSplit trainData, testData;
    private static ImageTransform imageTransform;
    private static DataNormalization scaler = new ImagePreProcessingScaler(0,1);

    public static void setup() throws IOException {

        File input = new ClassPathResource("HorseBreed").getFile();

        // Complete this method by adding codes for file split and input split
        // Note: You would also need to use a balanced path filter

        /*
         *
         * Write your codes here
         *
         */

        InputSplit[] allData = /* Write your codes here*/;
        trainData = allData[0];
        testData = allData[1];
    }

    public static DataSetIterator makeIterator(boolean train) throws IOException {

        ImageRecordReader rr = new ImageRecordReader(height,width,nChannel,labelMaker);

        // Complete this method by
        // 1. initializing your record reader both for train and test,
        // 2. creating an iterator
        // 3. perform normalization with ImagePreProcessingScaler, scaler
        // Note: remember you also want to do some image augmentations

        /*
         *
         * Write your codes here
         *
         */

        return null;
    }

    public static DataSetIterator getTrain(ImageTransform transform, int batchsize) throws IOException {
        imageTransform = transform;
        batchSize = batchsize;
        return makeIterator(true);
    }

    public static DataSetIterator getTest(int batchsize) throws IOException {
        batchSize = batchsize;
        return makeIterator(false);
    }
}
