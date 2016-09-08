package org.apache.ctakes.neural;

import java.io.File;
import java.io.IOException;

import org.cleartk.ml.Feature;
import org.cleartk.ml.encoder.CleartkEncoderException;
import org.cleartk.ml.encoder.features.FeaturesEncoder_ImplBase;
import org.cleartk.ml.util.featurevector.FeatureVector;

public class NoopFeaturesEncoder extends
    FeaturesEncoder_ImplBase<FeatureVector, String> {

  @Override
  public FeatureVector encodeAll(Iterable<Feature> features)
      throws CleartkEncoderException {
    return null;
  }

  @Override
  public void finalizeFeatureSet(File outputDirectory) throws IOException {
    super.finalizeFeatureSet(outputDirectory);
  }
}
