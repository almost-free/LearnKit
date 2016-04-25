//
//  LNKHillClimbingSearch.h
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKOptimization.h"

NS_ASSUME_NONNULL_BEGIN

/// A hill climbing search implementation that tries to maximize the given function by optimizing the initial set of parameters.
@interface LNKHillClimbingSearch : NSOperation

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

- (instancetype)initWithFunction:(LNKMultivariateFunction)function parameters:(LNKVector)parameters stepSizes:(LNKVector)stepSizes minParameter:(LNKVector)minParameters maxParameters:(LNKVector)maxParameters;

@property (readonly) LNKFloat optimalY;

@end

NS_ASSUME_NONNULL_END
