//
//  LNKLogRegClassifier.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKLogRegClassifier.h"

#import "_LNKLogRegClassifierLBFGS_AC.h"
#import "LNKMatrix.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

@implementation LNKLogRegClassifier {
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
	if (![algorithm isKindOfClass:[LNKOptimizationAlgorithmLBFGS class]]) {
		@throw [NSException exceptionWithName:NSInternalInconsistencyException
									   reason:@"For logistic regression classifiers, the only supported algorithm is L-BFGS."
									 userInfo:nil];
	}
	
	if (implementation == LNKImplementationTypeAccelerate) {
		return [_LNKLogRegClassifierLBFGS_AC class];
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
