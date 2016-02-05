//
//  LNKDiscreteProbabilityDistribution.h
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKClassProbabilityDistribution.h"

NS_ASSUME_NONNULL_BEGIN

@interface LNKDiscreteProbabilityDistribution : LNKClassProbabilityDistribution

/// Laplacian Smoothing should be enabled whenever probabilities can be zero to prevent calculation
/// errors. The default is `YES`.
@property (nonatomic) BOOL performsLaplacianSmoothing;

/// This is only applicable when `performsLaplacianSmoothing` is set to `YES`. The default is 1.
@property (nonatomic) NSUInteger laplacianSmoothingFactor;

/// Prior to building the distribution, all possible value types must be registered for each column.
- (void)registerValues:(NSArray<NSNumber *> *)values forColumnAtIndex:(LNKSize)columnIndex;

@end

NS_ASSUME_NONNULL_END
