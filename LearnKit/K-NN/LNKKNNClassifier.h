//
//  LNKKNNClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

/// The optimization algorithm for k-NN classifiers is ignored and can be `nil`.
/// Predicted values are of type LNKClass.
@interface LNKKNNClassifier : LNKClassifier

/// The value of `k` must be >= 1 and less than the number of examples in the design matrix.
@property (nonatomic) LNKSize k;

@end
