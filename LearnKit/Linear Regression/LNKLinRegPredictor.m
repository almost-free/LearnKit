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

- (Class)_classForImplementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm {
	if (implementation == LNKImplementationTypeAccelerate) {
		if ([algorithm isKindOfClass:[LNKOptimizationAlgorithmGradientDescent class]]) {
			return [_LNKLinRegPredictorGD_AC class];
		}
		else if ([algorithm isKindOfClass:[LNKOptimizationAlgorithmNormalEquations class]]) {
			return [_LNKLinRegPredictorNE_AC class];
		}
		else if ([algorithm isKindOfClass:[LNKOptimizationAlgorithmLBFGS class]]) {
			return [_LNKLinRegPredictorLBFGS_AC class];
		}
	}
	
	NSAssertNotReachable(@"Unsupported implementation type / algorithm", nil);
	
	return Nil;
}

- (LNKFloat *)_thetaVector {
	return _thetaVector;
}

- (void)_setThetaVector:(const LNKFloat *)thetaVector {
	NSParameterAssert(thetaVector);
	LNKFloatCopy(_thetaVector, thetaVector, self.matrix.columnCount);
}

@end
