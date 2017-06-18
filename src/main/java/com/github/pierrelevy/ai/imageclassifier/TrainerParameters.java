 /* Copyright 2017 Pierre Levy
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

/**
 * TrainerParameters
 */
public class TrainerParameters
{

   private String pathData;
    private int epochs;
    private int numLabels;
    private int numExamples;
    private int batchSize;
    private double splitTrainTest;
    private int listenerFreq;
    private int nCores;
    private boolean enableUIServer;

    /**
     * @return the pathData
     */
    public String getPathData()
    {
        return pathData;
    }

    /**
     * @param pathData the pathData to set
     */
    public void setPathData(String pathData)
    {
        this.pathData = pathData;
    }
     /**
     * @return the epochs
     */
    public int getEpochs()
    {
        return epochs;
    }

    /**
     * @param epochs the epochs to set
     */
    public void setEpochs(int epochs)
    {
        this.epochs = epochs;
    }

    /**
     * @return the numLabels
     */
    public int getNumLabels()
    {
        return numLabels;
    }

    /**
     * @param numLabels the numLabels to set
     */
    public void setNumLabels(int numLabels)
    {
        this.numLabels = numLabels;
    }

    /**
     * @return the numExamples
     */
    public int getNumExamples()
    {
        return numExamples;
    }

    /**
     * @param numExamples the numExamples to set
     */
    public void setNumExamples(int numExamples)
    {
        this.numExamples = numExamples;
    }

    /**
     * @return the batchSize
     */
    public int getBatchSize()
    {
        return batchSize;
    }

    /**
     * @param batchSize the batchSize to set
     */
    public void setBatchSize(int batchSize)
    {
        this.batchSize = batchSize;
    }

    /**
     * @return the splitTrainTest
     */
    public double getSplitTrainTest()
    {
        return splitTrainTest;
    }

    /**
     * @param splitTrainTest the splitTrainTest to set
     */
    public void setSplitTrainTest(double splitTrainTest)
    {
        this.splitTrainTest = splitTrainTest;
    }

    /**
     * @return the listenerFreq
     */
    public int getListenerFreq()
    {
        return listenerFreq;
    }

    /**
     * @param listenerFreq the listenerFreq to set
     */
    public void setListenerFreq(int listenerFreq)
    {
        this.listenerFreq = listenerFreq;
    }

    /**
     * @return the nCores
     */
    public int getNumCores()
    {
        return nCores;
    }

    /**
     * @param nCores the nCores to set
     */
    public void setnCores(int nCores)
    {
        this.nCores = nCores;
    }

    public boolean isUIServerEnabled()
    {
        return this.enableUIServer;
    }
    
    public void setUIServer( boolean enabled )
    {
        this.enableUIServer = enabled;
    }
 
}
