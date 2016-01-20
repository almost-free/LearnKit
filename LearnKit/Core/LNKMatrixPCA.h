//
//  LNKMatrixPCA.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKMatrix.h"

@interface LNKMatrix (PCA)

/// The dimension must be at least 1 and less than the number of columns in the matrix.
- (nullable LNKMatrix *)matrixReducedToDimension:(LNKSize)dimension;

@end
