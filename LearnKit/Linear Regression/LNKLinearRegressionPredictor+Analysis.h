//
//  LNKLinearRegressionPredictor+Analysis.h
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKLinearRegressionPredictor.h"

NS_ASSUME_NONNULL_BEGIN

@interface LNKLinearRegressionPredictor (Analysis)

/// The returned vector is +1 reference counted.
- (LNKVector)computeResiduals;
- (LNKVector)computeStandardizedResiduals;

- (LNKFloat)computeAIC;
- (LNKFloat)computeBIC;

- (LNKFloat)computeR2;

@property (nonatomic, readonly) LNKMatrix *hatMatrix;

@end

NS_ASSUME_NONNULL_END
