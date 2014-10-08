//
//  LNKKNNClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

typedef LNKFloat(^LNKKNNDistanceFunction)(const LNKFloat *example1, const LNKFloat *example2, LNKSize n);
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
