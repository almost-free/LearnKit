//
//  LNKNaiveBayesClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

NS_ASSUME_NONNULL_BEGIN

@class LNKClassProbabilityDistribution;

/// The optimization algorithm for Naive Bayes classifiers is ignored and can be `nil`.
/// Conditional independence is assumed.
/// Predicted values are of type LNKClass.
@interface LNKNaiveBayesClassifier : LNKClassifier

- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementation optimizationAlgorithm:(nullable id<LNKOptimizationAlgorithm>)algorithm classes:(LNKClasses *)classes NS_UNAVAILABLE;
- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementation optimizationAlgorithm:(nullable id<LNKOptimizationAlgorithm>)algorithm classes:(LNKClasses *)classes probabilityDistribution:(LNKClassProbabilityDistribution *)probabilityDistribution NS_DESIGNATED_INITIALIZER;

@property (nonatomic, retain, readonly, nonnull) __kindof LNKClassProbabilityDistribution *probabilityDistribution;

- (id)predictValueForFeatureVector:(LNKVector)featureVector probability:(nullable LNKFloat *)outProbability;

@end

NS_ASSUME_NONNULL_END
