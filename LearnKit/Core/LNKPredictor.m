//
//  LNKPredictor.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKPredictor.h"

#import "LNKDesignMatrix.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

@implementation LNKPredictor

- (instancetype)init {
	NSAssertNotReachable(@"Use the designated initializer", nil);
	return nil;
}

- (instancetype)initWithDesignMatrix:(LNKDesignMatrix *)matrix optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm {
	NSParameterAssert(matrix);
	
	if (!(self = [super init]))
		return nil;
	
	_designMatrix = [matrix retain];
	_algorithm = [algorithm retain];
	
	return self;
}

- (instancetype)initWithDesignMatrix:(LNKDesignMatrix *)matrix implementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm {
	NSAssert(![self isMemberOfClass:[LNKPredictor class]], @"Use a concrete subclass of LNKPredictor");
	
	Class class = [self _classForImplementationType:implementation optimizationAlgorithm:algorithm];
	return [[class alloc] initWithDesignMatrix:matrix optimizationAlgorithm:algorithm];
}

- (Class)_classForImplementationType:(LNKImplementationType)implementation {
#pragma unused(implementation)
	
	NSAssertNotReachable(@"%s should be implemented by subclasses", __PRETTY_FUNCTION__);
	return Nil;
}

- (void)dealloc {
	[_designMatrix release];
	[_algorithm release];
	[super dealloc];
}

- (void)train {
	NSAssertNotReachable(@"%s should be implemented by subclasses", __PRETTY_FUNCTION__);
}

- (id)predictValueForFeatureVector:(const LNKFloat *)featureVector length:(LNKSize)length {
#pragma unused(featureVector)
#pragma unused(length)
	
	NSAssertNotReachable(@"%s should be implemented by subclasses", __PRETTY_FUNCTION__);
	return nil;
}

@end
