package org.apache.ctakes.neural;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang.NotImplementedException;
import org.cleartk.ml.Feature;
import org.cleartk.ml.encoder.CleartkEncoderException;
import org.cleartk.ml.encoder.features.FeaturesEncoder_ImplBase;
import org.cleartk.ml.util.featurevector.FeatureVector;

public class NoopFeaturesEncoder extends
    FeaturesEncoder_ImplBase<FeatureVector, String> {

  @Override
  public FeatureVector encodeAll(Iterable<Feature> features)
      throws CleartkEncoderException {
    throw new NotImplementedException();
  }

  @Override
  public void finalizeFeatureSet(File outputDirectory) throws IOException {
    super.finalizeFeatureSet(outputDirectory);
  }
  
//  @Override
//  public String[] encodeAll(Iterable<Feature> features)
//      throws CleartkEncoderException {
//    List<Feature> featureList = new ArrayList<>();
//    for(Feature feature : features){
//      featureList.add(feature);
//    }
//    
//    String[] out = new String[featureList.size()];
//    
//    for(int i = 0; i < out.length; i++){
//      out[i] = featureList.get(i).getValue().toString();
//    }
//    return out;
//  }

}
