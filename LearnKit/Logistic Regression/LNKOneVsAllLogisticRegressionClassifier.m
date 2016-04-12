//
//  LNKOneVsAllLogisticRegressionClassifier.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKOneVsAllLogisticRegressionClassifier.h"

#import "_LNKOneVsAllLogisticRegressionClassifierLBFGS_AC.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKRegularizationConfiguration.h"

@implementation LNKOneVsAllLogisticRegressionClassifier

+ (NSArray<NSNumber *> *)supportedImplementationTypes {
	return @[ @(LNKImplementationTypeAccelerate) ];
}

+ (NSArray<Class> *)supportedAlgorithms {
	return @[ [LNKOptimizationAlgorithmLBFGS class] ];
}

+ (Class)_classForImplementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(Class)algorithm {
#pragma unused(implementationType)
#pragma unused(algorithm)
	
	return [_LNKOneVsAllLogisticRegressionClassifierLBFGS_AC class];
}

- (void)dealloc {
	[_regularizationConfiguration release];
	[super dealloc];
}

@end
