//
//  LNKCollaborativeFilteringPredictor.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKPredictor.h"

/// A collaborative filtering predictor for recommendation engines.
/// Supported optimization algorithms:
///  - Conjugate Gradient/Accelerate
@interface LNKCollaborativeFilteringPredictor : LNKPredictor

/// Initializes a new collaborative filtering predictor.
/// The `userCount` must be greater than 0.
- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm userCount:(NSUInteger)userCount;

/// Both the indicator and output matrices must be set prior to training the predictor.
/// The indicator and output matrices must be of `exampleCount` * `userCount` size.
/// A copy of the input matrix will be made.
- (void)copyIndicatorMatrix:(const LNKFloat *)matrix shouldTranspose:(BOOL)shouldTranspose;
- (void)copyOutputMatrix:(const LNKFloat *)matrix shouldTranspose:(BOOL)shouldTranspose;

@end
