//
//  LNKLinRegPredictor.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKPredictor.h"

/// For linear regression predictors, the gradient descent, normal equations, and L-BFGS algorithms are supported.
/// Regularization is only supported for gradient descent and L-BFGS algorithms.
/// Predicted values are of type NSNumber / LNKFloat.
@interface LNKLinRegPredictor : LNKPredictor

@end
