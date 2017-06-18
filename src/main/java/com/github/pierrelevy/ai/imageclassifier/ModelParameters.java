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
 * ModelParameters
 */
public class ModelParameters 
{

    private int height;
    private int width;
    private int channels;
    private int iterations;
    private long seed;

    /**
     * @return the height
     */
    public int getHeight()
    {
        return height;
    }

    /**
     * @param height the height to set
     */
    public void setHeight(int height)
    {
        this.height = height;
    }

    /**
     * @return the width
     */
    public int getWidth()
    {
        return width;
    }

    /**
     * @param width the width to set
     */
    public void setWidth(int width)
    {
        this.width = width;
    }

    /**
     * @return the channels
     */
    public int getChannels()
    {
        return channels;
    }

    /**
     * @param channels the channels to set
     */
    public void setChannels(int channels)
    {
        this.channels = channels;
    }


    /**
     * @return the seed
     */
    public long getSeed()
    {
        return seed;
    }

    /**
     * @param seed the seed to set
     */
    public void setSeed(long seed)
    {
        this.seed = seed;
    }

    /**
     * @return the iterations
     */
    public int getIterations()
    {
        return iterations;
    }

    /**
     * @param iterations the iterations to set
     */
    public void setIterations(int iterations)
    {
        this.iterations = iterations;
    }
}
