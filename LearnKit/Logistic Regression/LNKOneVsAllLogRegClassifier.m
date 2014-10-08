//
//  LNKOneVsAllLogRegClassifier.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKOneVsAllLogRegClassifier.h"

#import "_LNKOneVsAllLogRegClassifierLBFGS_AC.h"
#import "LNKOptimizationAlgorithm.h"

@implementation LNKOneVsAllLogRegClassifier

- (Class)_classForImplementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm {
	if (![algorithm isKindOfClass:[LNKOptimizationAlgorithmLBFGS class]]) {
		@throw [NSException exceptionWithName:NSInternalInconsistencyException
									   reason:@"For logistic regression classifiers, the only supported algorithm is L-BFGS."
									 userInfo:nil];
	}
	
	if (implementation == LNKImplementationTypeAccelerate)
		return [_LNKOneVsAllLogRegClassifierLBFGS_AC class];
	
	NSAssertNotReachable(@"Unsupported implementation type / algorithm", nil);
	
	return Nil;
}

@end
