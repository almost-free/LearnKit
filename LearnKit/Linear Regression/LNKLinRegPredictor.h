//
//  LNKLinRegPredictor.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKPredictor.h"

NS_ASSUME_NONNULL_BEGIN

/// For linear regression predictors, the gradient descent, normal equations, and L-BFGS algorithms are supported.
/// Regularization is only supported for gradient descent and L-BFGS algorithms.
/// Predicted values are of type NSNumber / LNKFloat.
/// A bias column is added to the matrix automatically.
@interface LNKLinRegPredictor : LNKPredictor
@end

NS_ASSUME_NONNULL_END
