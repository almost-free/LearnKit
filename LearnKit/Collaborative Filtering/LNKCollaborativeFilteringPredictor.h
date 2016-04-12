//
//  LNKCollaborativeFilteringPredictor.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKPredictor.h"

NS_ASSUME_NONNULL_BEGIN

@class LNKRegularizationConfiguration;

/// A collaborative filtering predictor for recommendation engines.
/// Supported optimization algorithms:
///  - Conjugate Gradient/Accelerate
@interface LNKCollaborativeFilteringPredictor : LNKPredictor

/// Initializes a new collaborative filtering predictor.
/// The `featureCount` must be greater than 0.
/// The matrix passed in must be the output matrix, with dimensions `rowCount` * `userCount`.
/// The indicator matrix must also be provided, with dimensions `rowCount` * `userCount`.
- (instancetype)initWithMatrix:(LNKMatrix *)outputMatrix indicatorMatrix:(LNKMatrix *)indicatorMatrix implementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm featureCount:(NSUInteger)featureCount;

@property (nonatomic, retain, nullable) LNKRegularizationConfiguration *regularizationConfiguration;

/// Load in pre-trained data and theta matrices.
/// The data matrix must be of dimensions `rowCount` * `featureCount`.
/// The theta matrix must be of dimensions `userCount` * `featureCount`.
- (void)loadDataMatrix:(LNKMatrix *)dataMatrix;
- (void)loadThetaMatrix:(LNKMatrix *)thetaMatrix;

/// Returns the top-k predictions for the given user.
/// `k` must be greater than 0.
- (NSIndexSet *)findTopK:(LNKSize)k predictionsForUser:(LNKSize)userIndex;

@end

NS_ASSUME_NONNULL_END
