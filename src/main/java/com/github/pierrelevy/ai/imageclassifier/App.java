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
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Properties;
import java.util.ResourceBundle;
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
    private static final String MODEL_BUNDLE = "conf/model.properties";
    private static final String TRAINER_BUNDLE = "conf/trainer.properties";

    static Logger log = Logger.getRootLogger();
    
    private static final String KEY_HEIGHT = "height";
    private static final String KEY_WIDTH = "width";
    private static final String KEY_CHANNELS = "channels";
    private static final int DEFAULT_HEIGHT = 100;
    private static final int DEFAULT_WIDTH = 100;
    private static final int DEFAULT_CHANNELS = 3;
    private static final long SEED = 42;

    private static final int iterations = 1;
    private static final String KEY_EPOCHS = "epochs";
    private static final String KEY_NUM_SAMPLES = "numSamples";
    private static final String KEY_BATCH_SIZE = "batchSize";
    private static final String KEY_DATA_PATH = "dataPath";
    private static final int DEFAULT_EPOCHS = 2;
    private static final int DEFAULT_NUM_SAMPLES = 10;
    private static final int DEFAULT_BATCH_SIZE = 20;
    private static final String DEFAULT_DATA_PATH = "data";
    private static final double SPLIT_TRAIN_TEST = 0.8;
    private static final int LISTENER_FREQUENCY = 1;
    private static final int NUM_CORE = 2;
    private static final boolean save = true;
    private static final boolean uiserver = false;
    

    public static void main(String[] args) throws IOException
    {
        if( args.length < 2 )
        {
           error();
        }

        
        String modelPath = FilenameUtils.concat(System.getProperty( "user.dir" ), "src/main/resources/" ) + "model.bin";
        Properties modelProperties = getProperties( MODEL_BUNDLE );
        Properties trainerProperties = getProperties( TRAINER_BUNDLE );
        ModelParameters modelParameters = loadModelParameters( modelProperties );

        if( args[0].equals( "train" ) )
        {
            String dataPath = args[1];
            int numLabels = new File( dataPath).listFiles().length;

            TrainerParameters trainerParameters = loadTrainerParameter( trainerProperties , numLabels );
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
    
    private static ModelParameters loadModelParameters( Properties modelProperties ) throws IOException
    {        
        ModelParameters parameters = new ModelParameters();
        // Model parameters values
        
        parameters.setHeight( getInt( modelProperties , KEY_HEIGHT , DEFAULT_HEIGHT ));
        parameters.setWidth( getInt( modelProperties , KEY_WIDTH , DEFAULT_WIDTH ));
        parameters.setChannels( getInt( modelProperties , KEY_CHANNELS , DEFAULT_CHANNELS ));
        parameters.setIterations(iterations);
        parameters.setSeed(SEED);
        
        return parameters;

    }
    
    private static Properties getProperties( String strFile ) throws FileNotFoundException, IOException
    {
        Properties properties = new Properties();
        properties.load( new FileInputStream( strFile ));
        return properties;        
    }
    
    private static String getString( Properties properties , String key, String defaultValue )
    {
        return properties.getProperty( key , defaultValue );
    }

    private static int getInt( Properties properties , String key, int defaultValue )
    {
        int nReturn = defaultValue;
        try
        {
            nReturn = Integer.parseInt( properties.getProperty( key ));
        }
        catch( NumberFormatException e )
        {
            log.error( "Error reading configuration key :" + key );
        }
        
        return nReturn;
    }
    
    private static TrainerParameters loadTrainerParameter( Properties trainerProperties , int numLabels ) throws IOException
    {
        TrainerParameters parameters = new TrainerParameters();
        
        // Trainers parameters values
        parameters.setPathData( getString( trainerProperties , KEY_DATA_PATH , DEFAULT_DATA_PATH ));
        parameters.setNumExamples( getInt( trainerProperties , KEY_NUM_SAMPLES , DEFAULT_NUM_SAMPLES ) );
        parameters.setBatchSize( getInt( trainerProperties , KEY_BATCH_SIZE , DEFAULT_BATCH_SIZE ) );
        parameters.setEpochs( getInt( trainerProperties , KEY_EPOCHS , DEFAULT_EPOCHS ) );
        parameters.setListenerFreq( LISTENER_FREQUENCY );
        parameters.setSplitTrainTest( SPLIT_TRAIN_TEST );
        parameters.setnCores( NUM_CORE );
        parameters.setUIServer( uiserver );
        parameters.setNumLabels( numLabels );
        
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
