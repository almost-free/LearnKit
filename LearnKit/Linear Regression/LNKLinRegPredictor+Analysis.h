//
//  LNKLinRegPredictor+Analysis.h
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKLinRegPredictor.h"

NS_ASSUME_NONNULL_BEGIN

@interface LNKLinRegPredictor (Analysis)

/// The returned vector must be freed with `LNKVectorFree`.
- (LNKVector)computeResiduals;
- (LNKVector)computeStandardizedResiduals;

- (LNKFloat)computeAIC;
- (LNKFloat)computeBIC;

@property (nonatomic, readonly) LNKMatrix *hatMatrix;

@end

NS_ASSUME_NONNULL_END
