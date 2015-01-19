//
//  LNKClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClasses.h"
#import "LNKPredictor.h"

/// Abstract
@interface LNKClassifier : LNKPredictor

/// The desired output classes must be non-`nil`.
- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm classes:(LNKClasses *)classes NS_DESIGNATED_INITIALIZER;

@property (nonatomic, retain, readonly) LNKClasses *classes;

/// The classifier should be trained prior to calling these methods.
- (LNKFloat)computeClassificationAccuracyOnTrainingMatrix;
- (LNKFloat)computeClassificationAccuracyOnMatrix:(LNKMatrix *)matrix;

@end
