//
//  LNKClassProbabilityDistributionPrivate.h
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKClassProbabilityDistribution.h"

NS_ASSUME_NONNULL_BEGIN

@interface LNKClassProbabilityDistribution (Private)

- (void)_setPrior:(LNKFloat)prior forClassAtIndex:(LNKSize)index;

@end

NS_ASSUME_NONNULL_END
