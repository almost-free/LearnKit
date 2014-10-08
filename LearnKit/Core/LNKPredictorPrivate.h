//
//  LNKPredictorPrivate.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKPredictor.h"

@interface LNKPredictor (Private)

- (instancetype)initWithMatrix:(LNKMatrix *)matrix optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm;

/* For subclasses to override: */

- (Class)_classForImplementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm;
- (LNKFloat)_evaluateCostFunction;
- (void)train;
- (id)predictValueForFeatureVector:(const LNKFloat *)featureVector length:(LNKSize)length;

@end
