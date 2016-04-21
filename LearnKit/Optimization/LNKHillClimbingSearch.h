//
//  LNKHillClimbingSearch.h
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKOptimization.h"

NS_ASSUME_NONNULL_BEGIN

@interface LNKHillClimbingSearch : NSOperation

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

- (instancetype)initWithFunction:(LNKMultivariateFunction)function parameterCount:(LNKSize)parameterCount;

@property (readonly) LNKFloat optimalY;

@end

NS_ASSUME_NONNULL_END
