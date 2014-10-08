//
//  LNKKNNClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

typedef LNKFloat(^LNKKNNDistanceFunction)(const LNKFloat *example1, const LNKFloat *example2, LNKSize n);

extern const LNKKNNDistanceFunction LNKKNNEuclideanDistanceFunction;

/// The optimization algorithm for k-NN classifiers is ignored and can be `nil`.
/// Predicted values are of type LNKClass.
@interface LNKKNNClassifier : LNKClassifier

/// The value of `k` must be >= 1 and less than the number of examples in the design matrix.
/// The default value is 1.
@property (nonatomic) LNKSize k;

/// The distance function to use when comparing examples.
/// The default is `LNKKNNEuclideanDistanceFunction`.
@property (nonatomic, copy) LNKKNNDistanceFunction distanceFunction;

@end
