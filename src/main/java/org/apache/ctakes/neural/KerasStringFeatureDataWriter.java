package org.apache.ctakes.neural;

import java.io.File;
import java.io.FileNotFoundException;

import org.cleartk.ml.python.keras.KerasStringOutcomeClassifierBuilder;
import org.cleartk.ml.script.ScriptStringOutcomeDataWriter;

/**
 * <br>
 * @author Tim Miller
 * @version 2.0.1
 * 
 */
public class KerasStringFeatureDataWriter extends ScriptStringOutcomeDataWriter<KerasStringOutcomeClassifierBuilder>{

  public KerasStringFeatureDataWriter(File outputDirectory)
      throws FileNotFoundException {
    super(outputDirectory);
  }

  @Override
  protected KerasStringOutcomeClassifierBuilder newClassifierBuilder() {
    return new KerasStringOutcomeClassifierBuilder();
  }
}
