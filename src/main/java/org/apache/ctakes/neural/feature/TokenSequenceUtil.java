package org.apache.ctakes.neural.feature;

import org.apache.ctakes.typesystem.type.syntax.BaseToken;

public class TokenSequenceUtil {
  public static String tokenToString(BaseToken token){
    String lower = token.getCoveredText().toLowerCase();
    return lower.replace("\n", "<CR>").replace("\r", "<LF>");
  }

}
