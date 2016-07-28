package org.apache.ctakes.neural;

import java.io.File;
import java.io.IOException;

import org.apache.commons.lang.NotImplementedException;
import org.cleartk.ml.encoder.CleartkEncoderException;
import org.cleartk.ml.encoder.outcome.OutcomeEncoder;

public class NoopOutcomeEncoder implements OutcomeEncoder<String,Integer> {

  @Override
  public Integer encode(String outcome) throws CleartkEncoderException {
    throw new NotImplementedException();
  }

  @Override
  public String decode(Integer outcome) throws CleartkEncoderException {
    throw new NotImplementedException();
  }

  @Override
  public void finalizeOutcomeSet(File outputDirectory) throws IOException {
    File outcomes = new File(outputDirectory, "outcome-lookup.txt");
    if(!outcomes.exists()){
      outcomes.createNewFile();
    }
    return;
  }

}
