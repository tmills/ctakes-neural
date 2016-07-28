package org.apache.ctakes.neural;

import java.io.File;
import java.io.FileNotFoundException;

import org.cleartk.ml.CleartkProcessingException;
import org.cleartk.ml.util.featurevector.FeatureVector;

/**
 * <br>
 * @author Tim Miller
 * @version 2.0.1
 * 
 */
public class KerasStringFeatureDataWriter extends ScriptStringFeatureDataWriter<KerasStringFeatureClassifierBuilder>{

  public KerasStringFeatureDataWriter(File outputDirectory)
      throws FileNotFoundException {
    super(outputDirectory);
  }

  @Override
  protected KerasStringFeatureClassifierBuilder newClassifierBuilder() {
    return new KerasStringFeatureClassifierBuilder();
  }

  @Override
  protected void writeEncoded(FeatureVector features, Integer outcome)
      throws CleartkProcessingException {
    return;
  }
}
