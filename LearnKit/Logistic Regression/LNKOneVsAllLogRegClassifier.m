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

+ (NSArray *)supportedImplementationTypes {
	return @[ @(LNKImplementationTypeAccelerate) ];
}

+ (NSArray *)supportedAlgorithms {
	return @[ [LNKOptimizationAlgorithmLBFGS class] ];
}

+ (Class)_classForImplementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(Class)algorithm {
#pragma unused(implementationType)
#pragma unused(algorithm)
	
	return [_LNKOneVsAllLogRegClassifierLBFGS_AC class];
}

@end
