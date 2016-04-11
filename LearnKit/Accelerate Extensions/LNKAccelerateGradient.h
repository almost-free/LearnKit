//
//  LNKAccelerateGradient.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKMatrix.h"
#import "LNKOptimizationAlgorithm.h"

typedef void(^LNKHFunction)(LNKFloat *column, LNKSize m);
typedef LNKFloat(^LNKCostFunction)(const LNKFloat *theta);

/// Tries to minimize the cost function by adjusting the thetaVector using the gradient descent algorithm.
void LNK_learntheta_gd(LNKMatrix *matrix, LNKFloat *thetaVector, LNKOptimizationAlgorithmGradientDescent *algorithm, LNKCostFunction costFunction);

/// Tries to minimize the cost function by adjusting the thetaVector using the L-BFGS algorithm.
/// The value of `lambda` is ignored if regularization is disabled; all other parameters are required.
void LNK_learntheta_lbfgs(LNKMatrix *matrix, LNKFloat *thetaVector, BOOL regularizationEnabled, LNKFloat lambda, LNKHFunction hFunction, LNKCostFunction costFunction);
