//
//  _LNKLinearRegressionPredictorGD_AC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKLinearRegressionPredictorGD_AC.h"

#import "LNKAccelerateGradient.h"
#import "LNKLinearRegressionPredictorPrivate.h"
#import "LNKPredictorPrivate.h"
#import "LNKRegularizationConfiguration.h"

@implementation _LNKLinearRegressionPredictorGD_AC

- (void)train {
	LNK_learntheta_gd(self.matrix, [self _thetaVector], self.algorithm, self.regularizationConfiguration != nil, self.regularizationConfiguration.lambda, ^(const LNKFloat *thetaVector) {
#pragma unused(thetaVector)
		return [self _evaluateCostFunction];
	});
}

@end
