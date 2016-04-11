//
//  _LNKLinearRegressionPredictorGD_AC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKLinearRegressionPredictorGD_AC.h"

#import "LNKAccelerateGradient.h"
#import "LNKLinearRegressionPredictorPrivate.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

@implementation _LNKLinearRegressionPredictorGD_AC

- (void)train {
	NSAssert([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmGradientDescent class]], @"Unexpected algorithm");
	LNK_learntheta_gd(self.matrix, [self _thetaVector], self.algorithm, ^(const LNKFloat *thetaVector) {
#pragma unused(thetaVector)
		return [self _evaluateCostFunction];
	});
}

@end
