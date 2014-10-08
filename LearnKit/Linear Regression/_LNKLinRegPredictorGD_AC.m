//
//  _LNKLinRegPredictorGD_AC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKLinRegPredictorGD_AC.h"

#import "LNKAccelerateGradient.h"
#import "LNKLinRegPredictorPrivate.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

@implementation _LNKLinRegPredictorGD_AC

- (void)train {
	NSAssert([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmGradientDescent class]], @"Unexpected algorithm");
	LNK_learntheta_gd(self.matrix, [self _thetaVector], self.algorithm, ^(const LNKFloat *thetaVector) {
#pragma unused(thetaVector)
		return [self _evaluateCostFunction];
	});
}

@end
