//
//  LNKMatrixPrivate.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#define _EXAMPLE_IN_MATRIX_BUFFER(index) (matrixBuffer + (index) * columnCount)

NS_ASSUME_NONNULL_BEGIN

@interface LNKMatrix (Private)

- (LNKSize *)_shuffleIndices;

@end

NS_ASSUME_NONNULL_END
