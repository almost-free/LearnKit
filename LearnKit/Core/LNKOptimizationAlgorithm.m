//
//  LNKOptimizationAlgorithm.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKOptimizationAlgorithm.h"

@implementation LNKOptimizationAlgorithmNormalEquations
@end

@implementation LNKOptimizationAlgorithmRegularizable

- (instancetype)init {
	NSAssertNotReachable(@"LNKOptimizationAlgorithmRegularizable is an abstract class. Please use one of its subclasses.", nil);
	return nil;
}

- (instancetype)_init {
	return [super init];
}

- (void)setLambda:(LNKFloat)lambda {
	if (lambda < 0)
		[NSException raise:NSInvalidArgumentException format:@"A negative lambda value is not allowed"];
	
	[self willChangeValueForKey:@"lambda"];
	_lambda = lambda;
	[self didChangeValueForKey:@"lambda"];
}

- (BOOL)regularizationEnabled {
	return _lambda > 0;
}

@end

@implementation LNKOptimizationAlgorithmGradientDescent

+ (instancetype)algorithmWithAlpha:(LNKFloat)alpha iterationCount:(LNKSize)iterationCount {
	NSParameterAssert(iterationCount != NSNotFound);
	return [[[self alloc] _initWithAlpha:alpha iterationCount:iterationCount convergenceThreshold:0] autorelease];
}

+ (instancetype)algorithmWithAlpha:(LNKFloat)alpha convergenceThreshold:(LNKFloat)convergenceThreshold {
	NSParameterAssert(convergenceThreshold > 0);
	return [[[self alloc] _initWithAlpha:alpha iterationCount:NSNotFound convergenceThreshold:convergenceThreshold] autorelease];
}

- (instancetype)init {
	@throw [NSException exceptionWithName:NSGenericException reason:@"The designated initializer should be used" userInfo:nil];
}

- (instancetype)_initWithAlpha:(LNKFloat)alpha iterationCount:(LNKSize)iterationCount convergenceThreshold:(LNKFloat)convergenceThreshold {
	NSParameterAssert(alpha > 0);
	
	self = [super _init];
	if (self) {
		_alpha = alpha;
		_iterationCount = iterationCount;
		_convergenceThreshold = convergenceThreshold;
	}
	return self;
}

@end

@implementation LNKOptimizationAlgorithmStochasticGradientDescent

@end

@implementation LNKOptimizationAlgorithmLBFGS

- (instancetype)init {
	return [super _init];
}

@end

@implementation LNKOptimizationAlgorithmCG

- (instancetype)init {
	self = [super _init];
	if (self) {
		_iterationCount = 100;
	}
	return self;
}

@end
