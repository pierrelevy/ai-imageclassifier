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

import java.util.Date;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * TimeIterationListener
 */
public class TimeIterationListener implements IterationListener
{

    private static final long serialVersionUID = 1L;
    private boolean invoked;
    private long start;
    private int iterationCount;
    private static final Logger log = LoggerFactory.getLogger(TimeIterationListener.class );
    
    public TimeIterationListener( int iterationCount )
    {
        this.iterationCount = iterationCount;
        start = System.currentTimeMillis();
    }
    
    @Override
    public boolean invoked()
    {
        return invoked;
    }

    @Override
    public void invoke()
    {
        invoked = true;
    }

    @Override
    public void iterationDone(Model model, int iteration)
    {
        int currentIteration = iteration + 1;
        long elapsed = System.currentTimeMillis() - start;
        long remaining = (iterationCount - currentIteration) * elapsed / currentIteration;
        long minutes = remaining / ( 1000 * 60 );
        Date date = new Date( start + elapsed + remaining );
        log.info( "Remaining time : " + minutes + "mn - End expected : " + date.toString() );
    }

}
