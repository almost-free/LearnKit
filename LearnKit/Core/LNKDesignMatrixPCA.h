//
//  LNKDesignMatrixPCA.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKDesignMatrix.h"

@interface LNKDesignMatrix (PCA)

/// The dimension must be at least 1 and less than the number of columns in the design matrix.
- (LNKDesignMatrix *)matrixReducedToDimension:(LNKSize)dimension;

@end
