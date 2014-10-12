//
//  LNKLinRegPredictor.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKLinRegPredictor.h"

#import "_LNKLinRegPredictorGD_AC.h"
#import "_LNKLinRegPredictorNE_AC.h"
#import "_LNKLinRegPredictorLBFGS_AC.h"
#import "LNKMatrix.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

@implementation LNKLinRegPredictor {
	LNKFloat *_thetaVector;
}

+ (NSArray *)supportedImplementationTypes {
	return @[ @(LNKImplementationTypeAccelerate) ];
}

+ (NSArray *)supportedAlgorithms {
	return @[ [LNKOptimizationAlgorithmGradientDescent class], [LNKOptimizationAlgorithmNormalEquations class], [LNKOptimizationAlgorithmLBFGS class] ];
}

+ (Class)_classForImplementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(Class)algorithm {
#pragma unused(implementationType)
	
	if (algorithm == [LNKOptimizationAlgorithmGradientDescent class]) {
		return [_LNKLinRegPredictorGD_AC class];
	}
	else if (algorithm == [LNKOptimizationAlgorithmNormalEquations class]) {
		return [_LNKLinRegPredictorNE_AC class];
	}
	
	return [_LNKLinRegPredictorLBFGS_AC class];
}


- (instancetype)initWithMatrix:(LNKMatrix *)matrix optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm {
	self = [super initWithMatrix:matrix optimizationAlgorithm:algorithm];
	if (self) {
		// The Theta vector is initially zero.
		_thetaVector = LNKFloatCalloc(matrix.columnCount);
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
