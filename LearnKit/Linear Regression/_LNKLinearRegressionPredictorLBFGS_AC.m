//
//  _LNKLinearRegressionPredictorLBFGS_AC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKLinearRegressionPredictorLBFGS_AC.h"

#import "LNKAccelerateGradient.h"
#import "LNKLinearRegressionPredictorPrivate.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

@implementation _LNKLinearRegressionPredictorLBFGS_AC

- (void)train {
	NSAssert([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmLBFGS class]], @"Unexpected algorithm");
	
	LNKOptimizationAlgorithmLBFGS *algorithm = self.algorithm;
	LNKMatrix *matrix = self.matrix;
	LNKFloat *thetaVector = [self _thetaVector];
	const LNKSize columnCount = matrix.columnCount;
	
	LNK_learntheta_lbfgs(matrix, thetaVector, algorithm.regularizationEnabled, algorithm.lambda, NULL, ^LNKFloat(const LNKFloat *theta) {
		LNKFloatCopy(thetaVector, theta, columnCount);
		return [self _evaluateCostFunction];
	});
}

@end
