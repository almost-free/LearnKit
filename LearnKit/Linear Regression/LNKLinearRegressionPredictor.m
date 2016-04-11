//
//  LNKLinearRegressionPredictor.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKLinearRegressionPredictor.h"

#import "_LNKLinearRegressionPredictorGD_AC.h"
#import "_LNKLinearRegressionPredictorNE_AC.h"
#import "_LNKLinearRegressionPredictorLBFGS_AC.h"
#import "LNKMatrix.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

@implementation LNKLinearRegressionPredictor {
	LNKFloat *_thetaVector;
}

+ (NSArray<NSNumber *> *)supportedImplementationTypes {
	return @[ @(LNKImplementationTypeAccelerate) ];
}

+ (NSArray<Class> *)supportedAlgorithms {
	return @[ [LNKOptimizationAlgorithmGradientDescent class], [LNKOptimizationAlgorithmNormalEquations class], [LNKOptimizationAlgorithmLBFGS class] ];
}

+ (Class)_classForImplementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(Class)algorithm {
#pragma unused(implementationType)
	
	if (algorithm == [LNKOptimizationAlgorithmGradientDescent class]) {
		return [_LNKLinearRegressionPredictorGD_AC class];
	}
	else if (algorithm == [LNKOptimizationAlgorithmNormalEquations class]) {
		return [_LNKLinearRegressionPredictorNE_AC class];
	}
	
	return [_LNKLinearRegressionPredictorLBFGS_AC class];
}


- (instancetype)initWithMatrix:(LNKMatrix *)matrix optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm {
	if (matrix.hasBiasColumn) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"Bias columns are added to matrices automatically by LinearRegression classifiers." userInfo:nil];
	}

	LNKMatrix *const workingMatrix = matrix.matrixByAddingBiasColumn;

	self = [super initWithMatrix:workingMatrix optimizationAlgorithm:algorithm];
	if (self) {
		// The Theta vector is initially zero.
		_thetaVector = LNKFloatCalloc(workingMatrix.columnCount);
	}
	return self;
}

- (void)dealloc {
	free(_thetaVector);
	[super dealloc];
}

- (LNKFloat *)_thetaVector {
	return _thetaVector;
}

- (void)_setThetaVector:(const LNKFloat *)thetaVector {
	NSParameterAssert(thetaVector);
	LNKFloatCopy(_thetaVector, thetaVector, self.matrix.columnCount);
}

@end
