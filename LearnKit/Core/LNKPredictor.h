//
//  LNKPredictor.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

@class LNKMatrix;
@protocol LNKOptimizationAlgorithm;

/// Abstract
@interface LNKPredictor : NSObject

- (instancetype)init NS_UNAVAILABLE;

/// Not all optimization algorithms are supported for all predictors.
/// Check each predictor's documentation for more information.
- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm;

@property (nonatomic, readonly) LNKMatrix *matrix;
@property (nonatomic, readonly) id <LNKOptimizationAlgorithm> algorithm;

- (void)train;

/// The predictor should be trained prior to calling this method.
/// The type of object returned varies by predictor.
- (id)predictValueForFeatureVector:(const LNKFloat *)featureVector length:(LNKSize)length;

@end
