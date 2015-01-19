//
//  LNKMatrixTestExtras.h
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKMatrix.h"

@interface LNKMatrix (TestExtras)

- (LNKMatrix *)polynomialMatrixOfDegree:(LNKSize)maxDegree;
- (LNKMatrix *)pairwisePolynomialMatrixOfDegree:(LNKSize)maxDegree;

@end
