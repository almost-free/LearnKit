//
//  _LNKLinRegPredictorLBFGS_AC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKLinRegPredictorLBFGS_AC.h"

#import "LNKAccelerateGradient.h"
#import "LNKLinRegPredictorPrivate.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

@implementation _LNKLinRegPredictorLBFGS_AC

- (void)train {
	NSAssert([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmLBFGS class]], @"Unexpected algorithm");
	
	LNKOptimizationAlgorithmLBFGS *algorithm = self.algorithm;
	LNKDesignMatrix *designMatrix = self.designMatrix;
	LNKFloat *thetaVector = [self _thetaVector];
	const LNKSize columnCount = designMatrix.columnCount;
	
	LNK_learntheta_lbfgs(designMatrix, thetaVector, algorithm.regularizationEnabled, algorithm.lambda, NULL, ^LNKFloat(const LNKFloat *theta) {
		LNKFloatCopy(thetaVector, theta, columnCount);
		return [self _evaluateCostFunction];
	});
}

@end
