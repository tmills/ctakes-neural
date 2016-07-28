package org.apache.ctakes.neural;

import java.io.File;
import java.io.IOException;
import java.util.jar.JarInputStream;
import java.util.jar.JarOutputStream;

import org.apache.uima.util.Level;
import org.cleartk.ml.jar.JarStreams;
import org.cleartk.ml.script.ScriptStringOutcomeClassifierBuilder;

import com.google.common.io.Files;

public class KerasStringFeatureClassifierBuilder extends ScriptStringOutcomeClassifierBuilder<KerasStringFeatureClassifier> {
  
  @Override
  public void packageClassifier(File dir, JarOutputStream modelStream)
      throws IOException {
    super.packageClassifier(dir, modelStream);

    JarStreams.putNextJarEntry(modelStream, "outcome-lookup.txt", new File(dir,
        "outcome-lookup.txt"));

    int modelNum = 0;
    while (true) {
      File modelArchFile = new File(dir, getArchFilename(modelNum));
      File modelWeightsFile = new File(dir, getWeightsFilename(modelNum));
      if (!modelArchFile.exists())
        break;

      JarStreams.putNextJarEntry(modelStream, modelArchFile.getName(),
          modelArchFile.getAbsoluteFile());
      JarStreams.putNextJarEntry(modelStream, modelWeightsFile.getName(),
          modelWeightsFile.getAbsoluteFile());
      modelNum++;
    }
  }

  @Override
  protected void unpackageClassifier(JarInputStream modelStream)
      throws IOException {
    super.unpackageClassifier(modelStream);

    // create the model dir to unpack all the model files
    this.modelDir = Files.createTempDir();

    // grab the script dir from the manifest:
    this.scriptDir = new File(modelStream.getManifest().getMainAttributes()
        .getValue(SCRIPT_DIR_PARAM));

    extractFileToDir(modelDir, modelStream, "outcome-lookup.txt");

    int modelNum = 0;
    while (true) {
      String archFn = getArchFilename(modelNum);
      String wtsFn = getWeightsFilename(modelNum);

      try {
        if (!extractFileToDir(modelDir, modelStream, archFn))
          break;
        if (!extractFileToDir(modelDir, modelStream, wtsFn))
          break;
      } catch (IOException e) {
        logger.log(Level.WARNING,
            "Encountered the following exception: " + e.getMessage());
        break;
      }
      modelNum++;
    }
  }

  @Override
  protected KerasStringFeatureClassifier newClassifier() {
    return new KerasStringFeatureClassifier(this.featuresEncoder,
        this.outcomeEncoder, this.modelDir, this.scriptDir);
  }
  
  private static String getArchFilename(int num) {
    return "model_" + num + ".json";
  }

  private static String getWeightsFilename(int num) {
    return "model_" + num + ".h5";
  }
}
