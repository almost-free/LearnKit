//
//  LNKKNNClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

NS_ASSUME_NONNULL_BEGIN

typedef LNKFloat(^LNKKNNDistanceFunction)(LNKVector example1, LNKVector example2);
extern const LNKKNNDistanceFunction LNKKNNEuclideanDistanceFunction;

typedef NS_ENUM(NSUInteger, LNKKNNOutputFunction) {
	/// This output function picks the example with the most frequent class and returns instances of `LNKClass`.
	LNKKNNOutputFunctionMostFrequent,
	
	/// This output function finds the average of the k-closest output values.
	/// It returns NSNumbers with type LNKFloat.
	LNKKNNOutputFunctionAverage
};

/// The optimization algorithm for k-NN classifiers is ignored and can be `nil`.
/// Predicted values depend on the output function used.
/// LNKKNNClassifier makes a copy of the matrix it is created with and normalizes it so that all features are on the same scale.
@interface LNKKNNClassifier : LNKClassifier

/// The value of `k` must be >= 1 and less than the number of examples in the matrix.
/// The default value is 1.
@property (nonatomic) LNKSize k;

/// The distance function to use when comparing examples.
/// The default is `LNKKNNEuclideanDistanceFunction`.
@property (nonatomic, copy) LNKKNNDistanceFunction distanceFunction;

/// The output function determines the predicted value from k-nearest neighbors.
/// The default is `LNKKNNOutputFunctionMostFrequent`.
@property (nonatomic) LNKKNNOutputFunction outputFunction;

@end

NS_ASSUME_NONNULL_END
