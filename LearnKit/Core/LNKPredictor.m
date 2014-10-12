//
//  LNKPredictor.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKPredictor.h"

#import "LNKMatrix.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

@implementation LNKPredictor

+ (NSArray *)supportedAlgorithms {
	[NSException raise:NSGenericException format:@"%s should be implemented by subclasses", __PRETTY_FUNCTION__];
	return nil;
}

+ (NSArray *)supportedImplementationTypes {
	[NSException raise:NSGenericException format:@"%s should be implemented by subclasses", __PRETTY_FUNCTION__];
	return nil;
}

+ (Class)_classForImplementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(Class)algorithm {
#pragma unused(implementationType)
#pragma unused(algorithm)
	
	[NSException raise:NSGenericException format:@"%s should be implemented by subclasses", __PRETTY_FUNCTION__];
	return Nil;
}


- (instancetype)init {
	NSAssertNotReachable(@"Use the designated initializer", nil);
	return nil;
}

- (instancetype)initWithMatrix:(LNKMatrix *)matrix optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm {
	NSParameterAssert(matrix);
	
	if (!(self = [super init]))
		return nil;
	
	_matrix = [matrix retain];
	_algorithm = [algorithm retain];
	
	return self;
}

- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm {
	NSAssert(![self isMemberOfClass:[LNKPredictor class]], @"Use a concrete subclass of LNKPredictor");
	
	if (![[[self class] supportedImplementationTypes] containsObject:@(implementationType)]) {
		[NSException raise:NSGenericException format:@"Unsupported implementation type"];
		return nil;
	}
	
	NSArray *supportedAlgorithms = [[self class] supportedAlgorithms];
	
	if (supportedAlgorithms != nil && ![supportedAlgorithms containsObject:[algorithm class]]) {
		[NSException raise:NSGenericException format:@"Unsupported optimization algorithm"];
		return nil;
	}
	
	Class class = [[self class] _classForImplementationType:implementationType optimizationAlgorithm:[algorithm class]];
	return [[class alloc] initWithMatrix:matrix optimizationAlgorithm:algorithm];
}

- (void)dealloc {
	[_matrix release];
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
