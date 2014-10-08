//
//  LNKLinRegPredictorPrivate.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKLinRegPredictor.h"

@interface LNKLinRegPredictor (Private)

- (LNKFloat *)_thetaVector NS_RETURNS_INNER_POINTER;
- (void)_setThetaVector:(const LNKFloat *)thetaVector;

@end
