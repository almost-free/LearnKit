//
//  LNKOptimizationAlgorithm.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKOptimizationAlgorithm.h"

@implementation LNKFixedAlpha

+ (instancetype)withValue:(LNKFloat)value {
	return [[[self alloc] initWithValue:value] autorelease];
}

- (instancetype)initWithValue:(LNKFloat)value {
	if (value <= 0)
		[NSException raise:NSInvalidArgumentException format:@"The alpha value must be greater than 0"];
	
	if (!(self = [super init]))
		return nil;
	
	_value = value;
	
	return self;
}

@end

@implementation LNKDecayingAlpha

+ (instancetype)withFunction:(LNKDecayingAlphaFunction)function {
	return [[[self alloc] initWithFunction:function] autorelease];
}

- (instancetype)initWithFunction:(LNKDecayingAlphaFunction)function {
	if (!function)
		[NSException raise:NSInvalidArgumentException format:@"The function must not be nil"];
	
	if (!(self = [super init]))
		return nil;
	
	_function = [function copy];
	
	return self;
}

- (void)dealloc {
	[_function release];
	[super dealloc];
}

@end


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

+ (instancetype)algorithmWithAlpha:(id <LNKAlpha>)alpha iterationCount:(LNKSize)iterationCount {
	NSParameterAssert(iterationCount != NSNotFound);
	return [[[self alloc] _initWithAlpha:alpha iterationCount:iterationCount convergenceThreshold:0] autorelease];
}

+ (instancetype)algorithmWithAlpha:(id <LNKAlpha>)alpha convergenceThreshold:(LNKFloat)convergenceThreshold {
	NSParameterAssert(convergenceThreshold > 0);
	return [[[self alloc] _initWithAlpha:alpha iterationCount:NSNotFound convergenceThreshold:convergenceThreshold] autorelease];
}

- (instancetype)init {
	@throw [NSException exceptionWithName:NSGenericException reason:@"The designated initializer should be used" userInfo:nil];
}

- (instancetype)_initWithAlpha:(id <LNKAlpha>)alpha iterationCount:(LNKSize)iterationCount convergenceThreshold:(LNKFloat)convergenceThreshold {
	if (!alpha)
		[NSException raise:NSInvalidArgumentException format:@"The alpha parameter must be specified"];
	
	self = [super _init];
	if (self) {
		_alpha = [alpha retain];
		_iterationCount = iterationCount;
		_convergenceThreshold = convergenceThreshold;
	}
	return self;
}

- (void)dealloc {
	[_alpha release];
	[super dealloc];
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
