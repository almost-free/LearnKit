//
//  LNKCollaborativeFilteringPredictor.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKPredictor.h"

@interface LNKCollaborativeFilteringPredictor : LNKPredictor

/// The `userCount` must be greater than 0.
- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm userCount:(NSUInteger)userCount;

/// The indicator and output matrices must be of `exampleCount` * `userCount` size.
/// Both the indicator and output matrices must be set prior to training the predictor.
- (void)copyIndicatorMatrix:(const LNKFloat *)matrix shouldTranspose:(BOOL)shouldTranspose;
- (void)copyOutputMatrix:(const LNKFloat *)matrix shouldTranspose:(BOOL)shouldTranspose;

@end
