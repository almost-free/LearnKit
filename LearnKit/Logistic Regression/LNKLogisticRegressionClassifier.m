//
//  LNKLogisticRegressionClassifier.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKLogisticRegressionClassifier.h"

#import "_LNKLogisticRegressionClassifierLBFGS_AC.h"
#import "LNKMatrix.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"
#import "LNKRegularizationConfiguration.h"

@implementation LNKLogisticRegressionClassifier {
	LNKFloat *_thetaVector;
}

+ (NSArray<NSNumber *> *)supportedImplementationTypes {
	return @[ @(LNKImplementationTypeAccelerate) ];
}

+ (NSArray<Class> *)supportedAlgorithms {
	return @[ [LNKOptimizationAlgorithmLBFGS class] ];
}

+ (Class)_classForImplementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(Class)algorithm {
#pragma unused(implementationType)
#pragma unused(algorithm)
	
	return [_LNKLogisticRegressionClassifierLBFGS_AC class];
}


- (instancetype)initWithMatrix:(LNKMatrix *)matrix optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm {
	if (matrix.hasBiasColumn) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"Bias columns are added to matrices automatically by LogisticRegression classifiers." userInfo:nil];
	}

	LNKMatrix *const workingMatrix = [matrix matrixByAddingBiasColumn];

	self = [super initWithMatrix:workingMatrix optimizationAlgorithm:algorithm];
	if (self) {
		// The Theta vector is initially zero.
		_thetaVector = LNKFloatCalloc(workingMatrix.columnCount);
	}
	return self;
}

- (void)dealloc {
	free(_thetaVector);
	[_regularizationConfiguration release];
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
