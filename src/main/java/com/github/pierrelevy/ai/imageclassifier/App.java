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
import org.apache.commons.io.FilenameUtils;
import org.apache.log4j.Logger;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

/**
 * Image Recognition App
 *
 */
public class App
{

    static Logger log = Logger.getRootLogger();
    protected static int height = 100;
    protected static int width = 100;
    protected static int channels = 3;
    protected static long seed = 42;

    protected static int iterations = 1;
    protected static int epochs = 200;
    protected static int numExamples = 500;
    protected static int batchSize = 20;
    protected static double splitTrainTest = 0.8;
    protected static int listenerFreq = 1;
    protected static int nCores = 2;
    protected static boolean save = true;
    protected static boolean uiserver = false;
    

    public static void main(String[] args) throws IOException
    {
        if( args.length < 2 )
        {
           error();
        }
        
        String modelPath = FilenameUtils.concat(System.getProperty( "user.dir" ), "src/main/resources/" ) + "model.bin";
        ModelParameters modelParameters = loadModelParameters();

        if( args[0].equals( "train" ) )
        {
            String dataPath = args[1];
            int numLabels = new File( dataPath).listFiles().length;

            TrainerParameters trainerParameters = loadTrainerParameter( dataPath , numLabels );
            displayTrainerParameters( trainerParameters , numLabels );

            displayModelParameters( modelParameters );
            train(dataPath, modelPath , trainerParameters , modelParameters );
        }
        else if( args[0].equals( "predict" ) )
        {
            predict(modelPath , args[1] , modelParameters );
        } 
        else
        {
            error();
        }
        
    }
    
    private static void error( )
    {
            System.out.println( "Syntaxe : \ntrain <data root directory>\nor\npredict <image file>\n");
            System.exit( 1 );
    }

    public static void train( String dataPath , String modelPath , TrainerParameters trainerParameters , ModelParameters modelParameters ) throws IOException
    {
        
        MultiLayerNetwork network = Trainer.train( NetworkFactory.NETWORK_LENET , trainerParameters , modelParameters);
        
        if (save) {
            log.info("Save model....");
            ModelSerializer.writeModel(network, modelPath , true);
        }
        log.info("****************Example finished********************");
    }
    
    static void predict( String modelPath , String imagePath , ModelParameters parameters ) throws IOException
    {
        log.info("**************** Loading model ********************");
        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork( modelPath );
        File file = new File( imagePath );
        NativeImageLoader loader = new NativeImageLoader( parameters.getHeight(), parameters.getWidth(), parameters.getChannels() );
        INDArray image = loader.asMatrix(file);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);
        INDArray output = network.output(image);
        INDArray labels = network.getLabels();


        log.info("## The FILE CHOSEN WAS " + imagePath );
        log.info("## The Neural Nets Prediction ##");
        log.info("## list of probabilities per label ##");
        if( labels != null )
        {
            log.info( labels.toString() );
        }
        log.info("## List of Labels in Order## ");
        log.info( output.toString() );            

   
        
    }
    
    private static ModelParameters loadModelParameters()
    {
        ModelParameters parameters = new ModelParameters();
        // Model parameters values
        parameters.setHeight(height);
        parameters.setWidth(width);
        parameters.setChannels(channels);
        parameters.setIterations(iterations);
        parameters.setSeed(seed);
        
        return parameters;

    }
    
    private static TrainerParameters loadTrainerParameter( String dataPath , int numLabels )
    {
        TrainerParameters parameters = new TrainerParameters();
        
        // Trainers parameters values
        parameters.setPathData( dataPath );
        parameters.setNumLabels( numLabels );
        parameters.setBatchSize( batchSize );
        parameters.setEpochs( epochs );
        parameters.setListenerFreq( listenerFreq );
        parameters.setNumExamples( numExamples );
        parameters.setSplitTrainTest( splitTrainTest );
        parameters.setnCores( nCores );
        parameters.setUIServer( uiserver );
        
        return parameters;

    }
    
    private static void displayTrainerParameters( TrainerParameters parameters , int numLabels )
    {
        System.out.println( "=================================================================" );
        System.out.println( "                      Trainer parameters" );
        System.out.println( "=================================================================" );
        System.out.println( "Data Path                  : " + parameters.getPathData() );
        System.out.println( "Category count             : " + numLabels );
        System.out.println( "Sample count per category  : " + parameters.getNumExamples() );
        System.out.println( "Sample count per batch     : " + parameters.getBatchSize() );
        System.out.println( "Epoch count                : " + parameters.getEpochs() );
        int numIterations = (int) ((parameters.getEpochs() * parameters.getNumExamples() * 4 * numLabels / parameters.getBatchSize()) * parameters.getSplitTrainTest());
        System.out.println( "Iteration count            : " + numIterations );
        System.out.println( "=================================================================" );
    }
    
    private static void displayModelParameters( ModelParameters parameters )
    {
        System.out.println( "=================================================================" );
        System.out.println( "                      Model parameters" );
        System.out.println( "=================================================================" );
        System.out.println( "Image size                 : " + parameters.getWidth() + "x" + parameters.getHeight() );
        System.out.println( "Color channels count       : " + parameters.getChannels() );
        System.out.println( "=================================================================" );
    }

}
