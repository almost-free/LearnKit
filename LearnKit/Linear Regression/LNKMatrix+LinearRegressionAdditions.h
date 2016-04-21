//
//  LNKMatrix+LinearRegressionAdditions.h
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKMatrix.h"

NS_ASSUME_NONNULL_BEGIN

@interface LNKMatrix (LinearRegressionAdditions)

- (void)performBoxCoxTransformationWithLambda:(LNKFloat)lambda;

@end

NS_ASSUME_NONNULL_END
