//
//  LNKClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClasses.h"
#import "LNKPredictor.h"

NS_ASSUME_NONNULL_BEGIN

/// Abstract
@interface LNKClassifier : LNKPredictor

- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementation optimizationAlgorithm:(nullable id<LNKOptimizationAlgorithm>)algorithm classes:(LNKClasses *)classes NS_DESIGNATED_INITIALIZER;

@property (nonatomic, retain, readonly) LNKClasses *classes;

/// The classifier should be trained prior to calling these methods.
- (LNKFloat)computeClassificationAccuracyOnTrainingMatrix;
- (LNKFloat)computeClassificationAccuracyOnMatrix:(LNKMatrix *)matrix;

@end

NS_ASSUME_NONNULL_END
