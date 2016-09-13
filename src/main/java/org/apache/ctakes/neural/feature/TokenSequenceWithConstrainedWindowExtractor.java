package org.apache.ctakes.neural.feature;

import java.util.ArrayList;
import java.util.List;

import org.apache.ctakes.typesystem.type.syntax.BaseToken;
import org.apache.ctakes.typesystem.type.textsem.IdentifiedAnnotation;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.tcas.Annotation;
import org.cleartk.ml.Feature;
import org.cleartk.ml.feature.extractor.CleartkExtractorException;
import org.cleartk.ml.feature.extractor.FeatureExtractor2;
import static org.apache.ctakes.neural.feature.TokenSequenceUtil.tokenToString;

public class TokenSequenceWithConstrainedWindowExtractor<X extends IdentifiedAnnotation,Y extends Annotation> implements FeatureExtractor2<X,Y>{

  private int window = 5;
  
  public TokenSequenceWithConstrainedWindowExtractor(int windowSize) {
    this.window = windowSize;
  }
  
  @Override
  public List<Feature> extract(JCas view, X target, Y constraint)
      throws CleartkExtractorException {
    List<Feature> feats = new ArrayList<>();
    
    List<BaseToken> beforeTokens = JCasUtil.selectPreceding(BaseToken.class, target, this.window);
    List<BaseToken> sentenceTokens = JCasUtil.selectCovered(BaseToken.class, constraint);

    beforeTokens.retainAll(sentenceTokens);
    
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
    afterTokens.retainAll(sentenceTokens);
    
    for(BaseToken token : afterTokens){
      feats.add(new Feature(tokenToString(token)));
    }
    
    for(int i = afterTokens.size(); i < this.window; i++){
      feats.add(new Feature("OOB_AFTER_"+i));
    }

    return feats;
  }

}
