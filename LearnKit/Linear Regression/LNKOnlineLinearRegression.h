//
//  LNKOnlineLinearRegression.h
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKTypes.h"

NS_ASSUME_NONNULL_BEGIN

/// A fast online linear regression predictor that learns parameters as examples are fed.
/// While not as featured as LNKLinearRegressionPredictor, it is ideal for real-time systems.
@interface LNKOnlineLinearRegression : NSObject

@property (nonatomic, readonly) LNKFloat intercept;
@property (nonatomic, readonly) LNKFloat slope;

- (void)addExampleWithX:(LNKFloat)x y:(LNKFloat)y;

- (void)regress;

- (LNKFloat)predictYForX:(LNKFloat)x;

@end

NS_ASSUME_NONNULL_END
