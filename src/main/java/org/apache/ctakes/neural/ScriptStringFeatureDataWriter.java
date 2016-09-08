package org.apache.ctakes.neural;

import java.io.File;
import java.io.FileNotFoundException;
//import java.util.Locale;

import org.apache.uima.UimaContext;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.fit.factory.initializable.Initializable;
import org.apache.uima.resource.ResourceInitializationException;
import org.cleartk.ml.CleartkProcessingException;
import org.cleartk.ml.Feature;
import org.cleartk.ml.Instance;
import org.cleartk.ml.encoder.outcome.StringToIntegerOutcomeEncoder;
import org.cleartk.ml.jar.DataWriter_ImplBase;
import org.cleartk.ml.script.ScriptStringOutcomeClassifierBuilder;
import org.cleartk.ml.util.featurevector.FeatureVector;

public class ScriptStringFeatureDataWriter
  extends DataWriter_ImplBase<ScriptStringOutcomeClassifierBuilder, FeatureVector, String,Integer> implements Initializable {
  
  public static final String PARAM_SCRIPT_DIR = "DataWriterScriptDirectory";
  @ConfigurationParameter(name=PARAM_SCRIPT_DIR)
  public String dir;
  
  public ScriptStringFeatureDataWriter(File outputDirectory)
      throws FileNotFoundException {
    super(outputDirectory);
    this.setFeaturesEncoder(new NoopFeaturesEncoder());
    this.setOutcomeEncoder(new StringToIntegerOutcomeEncoder());
  }

  @Override
  public void write(Instance<String> instance)
      throws CleartkProcessingException {
    // do this for the side effect of creating an outcome-lookup.txt with all the possible outcomes
    this.classifierBuilder.getOutcomeEncoder().encode(instance.getOutcome());
    
    this.trainingDataWriter.print(instance.getOutcome());
    this.trainingDataWriter.print(" |");
    for (Feature feat : instance.getFeatures()) {
      this.trainingDataWriter.print(' ');
      this.trainingDataWriter.print(feat.getValue());  
    }
    this.trainingDataWriter.println();
  }

  @Override
  public void initialize(UimaContext context)
      throws ResourceInitializationException {
    this.dir = (String) context.getConfigParameterValue(PARAM_SCRIPT_DIR);
    this.classifierBuilder.setScriptDirectory(this.dir);
  }


  @Override
  protected void writeEncoded(FeatureVector features, Integer outcome)
      throws CleartkProcessingException {
    return;
  }

  @Override
  protected ScriptStringOutcomeClassifierBuilder newClassifierBuilder() {
    return new ScriptStringOutcomeClassifierBuilder();
  }
}
