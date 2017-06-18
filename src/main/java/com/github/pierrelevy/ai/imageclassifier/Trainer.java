 /* Copyright 2017 authors Skymind,Inc., Pierre Levy
  *
  *    Licensed under the Apache License, Version 2.0 (the "License");
  *    you may not use this file except in compliance with the License.
  *    You may obtain a copy of the License at
  *
  *        http://www.apache.org/licenses/LICENSE-2.0
  *
  *    Unless required by applicable law or agreed to in writing, software
  *    distributed under the License is distributed on an "AS IS" BASIS,
  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  *    See the License for the specific language governing permissions and
  *    limitations under the License.
  */


package com.github.pierrelevy.ai.imageclassifier;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Trainer
 */
public class Trainer 
{
    protected static final Logger log = LoggerFactory.getLogger( Trainer.class );

    public static MultiLayerNetwork train( int nModel, TrainerParameters trainerParams, ModelParameters modelParams ) throws IOException
    {
        log.info("Load data....");
        /**
         * cd Data Setup -> organize and limit data file paths: - mainPath =
         * path to image files - fileSplit = define basic dataset split with
         * limits on format - pathFilter = define additional file load filter to
         * limit size and balance batch content
         *
         */
        Random rng = new Random( modelParams.getSeed() );

        
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
//        File mainPath = new File(System.getProperty("user.dir"), "data/guitars/");
        File mainPath = new File( trainerParams.getPathData() );
        FileSplit fileSplit = new FileSplit( mainPath, NativeImageLoader.ALLOWED_FORMATS, rng );
//        BalancedPathFilter pathFilter = new BalancedPathFilter( rng, labelMaker, trainerParams.getNumExamples() , trainerParams.getNumLabels(), trainerParams.getBatchSize() );
        BalancedPathFilter pathFilter = new BalancedPathFilter( rng, labelMaker, trainerParams.getNumExamples() );

        /**
         * Data Setup -> train test split - inputSplit = define train and test
         * split
         *
         */
        InputSplit[] inputSplit = fileSplit.sample(pathFilter, trainerParams.getSplitTrainTest(), 1.0 - trainerParams.getSplitTrainTest() );
        InputSplit trainData = inputSplit[0];
        InputSplit testData = inputSplit[1];

        System.out.println( "Number of images in the trainData dataset : " + trainData.length() );
        System.out.println( "Number of images in the testData dataset : " + testData.length() );

        /**
         * Data Setup -> transformation - Transform = how to tranform images and
         * generate large dataset to train on
         *
         */
        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
//        ImageTransform colorTransform = new ColorConversionTransform(new Random(seed), COLOR_BGR2YCrCb);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]
        {
            flipTransform1, warpTransform, flipTransform2
        });

        /**
         * Data Setup -> normalization - how to normalize images and generate
         * large dataset to train on
         *
         */
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

        log.info("Build model....");
        
        MultiLayerNetwork network = NetworkFactory.getNetwork( NetworkFactory.NETWORK_AXELNET , modelParams , trainerParams.getNumLabels() );
        network.init();
        
        List<IterationListener> listeners = new ArrayList<>();
        listeners.add( new ScoreIterationListener( trainerParams.getListenerFreq() ));
        
        int numIterations = (int) ((trainerParams.getEpochs() * trainerParams.getNumExamples() * 4 * trainerParams.getNumLabels() / trainerParams.getBatchSize()) * trainerParams.getSplitTrainTest());
        listeners.add( new TimeIterationListener( numIterations ) );
        
        if( trainerParams.isUIServerEnabled() )
        {
             //Initialize the user interface backend
            UIServer uiServer = UIServer.getInstance();

            //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
            StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

            //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
            uiServer.attach(statsStorage);

            //Then add the StatsListener to collect this information from the network, as it trains
            listeners.add( new StatsListener(statsStorage) );
        }

        network.setListeners( listeners );

        /**
         * Data Setup -> define how to load data into net:
         *  - recordReader = the reader that loads and converts image data pass in inputSplit to initialize
         *  - dataIter = a generator that only loads one batch at a time into memory to save memory
         *  - trainIter = uses MultipleEpochsIterator to ensure model runs through the data for all epochs
         **/
        ImageRecordReader recordReader = new ImageRecordReader( modelParams.getHeight(), modelParams.getWidth(), modelParams.getChannels(), labelMaker);
        DataSetIterator dataIter;
        MultipleEpochsIterator trainIter;


        log.info("Train model....");
        // Train without transformations
        recordReader.initialize( trainData , null );
        int nNumLabels = recordReader.getLabels().size();
        System.out.println("nNumLabels : " + nNumLabels );
        for( int i = 0 ; i < nNumLabels ; i++ )
        {
            log.info( recordReader.getLabels().get(i) );
        }


        dataIter = new RecordReaderDataSetIterator(recordReader, trainerParams.getBatchSize(), 1, trainerParams.getNumLabels() );
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        trainIter = new MultipleEpochsIterator( trainerParams.getEpochs(), dataIter, trainerParams.getNumCores());
        
        network.fit(trainIter);

        // Train with transformations
        for (ImageTransform transform : transforms ) {
            System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
            recordReader.initialize( trainData, transform);
            dataIter = new RecordReaderDataSetIterator(recordReader, trainerParams.getBatchSize(), 1, trainerParams.getNumLabels() );
            scaler.fit(dataIter);
            dataIter.setPreProcessor(scaler);
            trainIter = new MultipleEpochsIterator(trainerParams.getEpochs(), dataIter, trainerParams.getNumCores() );
            network.fit(trainIter);
        }

        log.info("Evaluate model....");
        recordReader.initialize( testData );
        dataIter = new RecordReaderDataSetIterator(recordReader, trainerParams.getBatchSize(), 1, trainerParams.getNumLabels() );
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        Evaluation eval = network.evaluate(dataIter);
        log.info(eval.stats(true));

        return network;
    }

}
