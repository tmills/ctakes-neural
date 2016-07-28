package org.apache.ctakes.neural;

import java.io.File;
import java.util.List;

import org.cleartk.ml.Feature;
import org.cleartk.ml.encoder.CleartkEncoderException;
import org.cleartk.ml.encoder.features.FeaturesEncoder;
import org.cleartk.ml.encoder.outcome.OutcomeEncoder;
import org.cleartk.ml.script.ScriptStringOutcomeClassifier;
import org.cleartk.ml.util.featurevector.FeatureVector;

public class KerasStringFeatureClassifier extends ScriptStringOutcomeClassifier{

  public KerasStringFeatureClassifier(
      FeaturesEncoder<FeatureVector> featuresEncoder,
      OutcomeEncoder<String, Integer> outcomeEncoder, File modelDir,
      File scriptDir) {
    super(featuresEncoder, outcomeEncoder, modelDir, scriptDir);
  }

  @Override
  protected String featuresToString(List<Feature> features)
      throws CleartkEncoderException {
    StringBuilder buff = new StringBuilder();
    for(Feature feat : features){
      buff.append(' ');
      buff.append(feat.getValue());
    }
    return buff.substring(1);
  }
}
