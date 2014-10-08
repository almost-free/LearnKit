//
//  LNKDesignMatrixTestExtras.h
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKDesignMatrix.h"

@interface LNKDesignMatrix (TestExtras)

- (LNKDesignMatrix *)polynomialMatrixOfDegree:(LNKSize)maxDegree;
- (LNKDesignMatrix *)pairwisePolynomialMatrixOfDegree:(LNKSize)maxDegree;

@end
