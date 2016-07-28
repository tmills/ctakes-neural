package org.apache.ctakes.neural.feature;

import java.util.ArrayList;
import java.util.List;

import org.apache.ctakes.typesystem.type.syntax.BaseToken;
import org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.cleartk.ml.Feature;
import org.cleartk.ml.feature.extractor.CleartkExtractorException;
import org.cleartk.ml.feature.extractor.FeatureExtractor1;

public class TokensSequenceWithWindowExtractor implements FeatureExtractor1<IdentifiedAnnotation>{

  private int window;
  
  public TokensSequenceWithWindowExtractor(int numTokens){
    this.window = numTokens;
  }
  
  @Override
  public List<Feature> extract(JCas view, IdentifiedAnnotation target)
      throws CleartkExtractorException {
    List<Feature> feats = new ArrayList<>();
    
    List<BaseToken> beforeTokens = JCasUtil.selectPreceding(BaseToken.class, target, this.window);
    
    for(int i = this.window; i > beforeTokens.size(); i--){
      feats.add(new Feature("OOB_BEFORE_" + i));
    }
    
    for(BaseToken token : beforeTokens){
      feats.add(new Feature(tokenToString(token)));
    }
    
    feats.add(new Feature("<e>"));
    for(BaseToken token : JCasUtil.selectCovered(BaseToken.class, target)){
      feats.add(new Feature(tokenToString(token)));
    }
    feats.add(new Feature("</e>"));
    
    List<BaseToken> afterTokens = JCasUtil.selectFollowing(BaseToken.class, target, this.window);
    
    for(BaseToken token : afterTokens){
      feats.add(new Feature(tokenToString(token)));
    }
    
    for(int i = afterTokens.size(); i < this.window; i++){
      feats.add(new Feature("OOB_AFTER_"+i));
    }
    
    return feats;
  }
  
  private static String tokenToString(BaseToken token){
    String lower = token.getCoveredText().toLowerCase();
    return lower.replace("\n", "<CR>").replace("\r", "<LF>");
  }
}
