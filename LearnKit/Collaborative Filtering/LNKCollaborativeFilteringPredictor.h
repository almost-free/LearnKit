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
/// The `featureCount` must be greater than 0.
/// The matrix passed in must be the output matrix, with dimensions `exampleCount` * `userCount`.
/// The indicator matrix must also be provided, with dimensions `exampleCount` * `userCount`.
- (instancetype)initWithMatrix:(LNKMatrix *)outputMatrix indicatorMatrix:(LNKMatrix *)indicatorMatrix implementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm featureCount:(NSUInteger)featureCount;

/// Load in pre-trained data and theta matrices.
/// The data matrix must be of dimensions `exampleCount` * `featureCount`.
/// The theta matrix must be of dimensions `userCount` * `featureCount`.
- (void)loadDataMatrix:(LNKMatrix *)dataMatrix;
- (void)loadThetaMatrix:(LNKMatrix *)thetaMatrix;

@end
