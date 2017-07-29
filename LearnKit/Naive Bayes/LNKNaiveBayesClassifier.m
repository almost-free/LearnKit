//
//  LNKNaiveBayesClassifier.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKNaiveBayesClassifier.h"

#import "_LNKNaiveBayesClassifierAC.h"
#import "LNKMatrix.h"

@implementation LNKNaiveBayesClassifier

+ (NSArray<NSNumber *> *)supportedImplementationTypes {
	return @[ @(LNKImplementationTypeAccelerate) ];
}

+ (NSArray<Class> *)supportedAlgorithms {
	return @[];
}

+ (Class)_classForImplementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(Class)algorithm {
#pragma unused(implementationType)
#pragma unused(algorithm)
	
	return [_LNKNaiveBayesClassifierAC class];
}

- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm classes:(LNKClasses *)classes probabilityDistribution:(LNKClassProbabilityDistribution *)probabilityDistribution {
	if (!(self = [super initWithMatrix:matrix implementationType:implementation optimizationAlgorithm:algorithm classes:classes]))
		return nil;

	_probabilityDistribution = [probabilityDistribution retain];

	return self;
}

- (void)dealloc {
	[_probabilityDistribution release];
	[super dealloc];
}

- (id)predictValueForFeatureVector:(LNKVector)featureVector probability:(LNKFloat *)outProbability {
#pragma unused(featureVector)
#pragma unused(outProbability)

	[NSException raise:NSInternalInconsistencyException format:@"%s must be overriden by subclasses", __PRETTY_FUNCTION__];
	return nil;
}

@end
