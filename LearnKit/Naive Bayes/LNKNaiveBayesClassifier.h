//
//  LNKNaiveBayesClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

NS_ASSUME_NONNULL_BEGIN

/// The optimization algorithm for Naive Bayes classifiers is ignored and can be `nil`.
/// All the features of the matrix should be categorical and represented as integers.
/// Conditional independence is assumed.
/// Predicted values are of type LNKClass.
@interface LNKNaiveBayesClassifier : LNKClassifier

/// To prevent underflow issues, the sum of the logarithms of probabilities can be maximized rather
/// than the product of probabilities. The default is `YES`.
@property (nonatomic) BOOL computesSumOfLogarithms;

/// Laplacian Smoothing should be enabled whenever probabilities can be zero to prevent calculation
/// errors. The default is `YES`.
@property (nonatomic) BOOL performsLaplacianSmoothing;

/// This is only applicable when `performsLaplacianSmoothing` is set to `YES`. The default is 1.
@property (nonatomic) NSUInteger laplacianSmoothingFactor;

/// Prior to training, all possible value types must be registered for each column.
- (void)registerValues:(NSArray<NSNumber *> *)values forColumn:(LNKSize)columnIndex;

- (id)predictValueForFeatureVector:(LNKVector)featureVector probability:(nullable LNKFloat *)outProbability;

@end

NS_ASSUME_NONNULL_END
