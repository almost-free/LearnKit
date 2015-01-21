//
//  LNKOptimizationAlgorithm.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKOptimizationAlgorithm.h"

#import "fmincg.h"
#import "LNKAccelerate.h"

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

- (void)runWithParameterVector:(LNKVector)vector exampleCount:(LNKSize)exampleCount delegate:(id<LNKOptimizationAlgorithmDelegate>)delegate {
	NSAssert([self.alpha isKindOfClass:[LNKFixedAlpha class]], @"Only fixed alpha values are suppored by this method");
	NSParameterAssert(vector.data);
	NSParameterAssert(vector.length);
	NSParameterAssert(delegate);
	NSParameterAssert(exampleCount);
	
	const LNKFloat alpha = [(LNKFixedAlpha *)self.alpha value];
	const LNKSize iterationCount = self.iterationCount;
	const LNKSize batchCount = self.stepCount == NSNotFound ? exampleCount : self.stepCount;
	const LNKSize batchSize = exampleCount / batchCount;
	
	LNKFloat *weights = LNKFloatAllocAndCopy(vector.data, vector.length);
	
	// Re-used across iterations.
	LNKFloat *gradient = LNKFloatAlloc(vector.length);
	
	for (LNKSize iteration = 0; iteration < iterationCount; iteration++) {
		for (NSUInteger batch = 0; batch < batchCount; batch++) {
			LNKRange range = LNKRangeMake(batch * batchSize, batch == batchCount - 1 ? exampleCount - batch * batchSize : batchSize);
			
			[delegate optimizationAlgorithmWillBeginIterationWithInputVector:weights];
			[delegate computeGradientForOptimizationAlgorithm:gradient inRange:range];
			
			// Multiply by alpha.
			LNK_vsmul(gradient, UNIT_STRIDE, &alpha, gradient, UNIT_STRIDE, vector.length);
			
			LNK_vsub(gradient, UNIT_STRIDE, weights, UNIT_STRIDE, weights, UNIT_STRIDE, vector.length);
		}
	}
	
	free(gradient);
}

@end

@implementation LNKOptimizationAlgorithmLBFGS

- (instancetype)init {
	return [super _init];
}

@end

@implementation LNKOptimizationAlgorithmCG {
	id<LNKOptimizationAlgorithmDelegate> _delegate;
	LNKSize _exampleCount;
}

- (instancetype)init {
	self = [super _init];
	if (self) {
		_iterationCount = 100;
	}
	return self;
}

static LNKOptimizationAlgorithmCG *tempSelf = nil;

static void _fmincg_evaluate(LNKFloat *inputVector, LNKFloat *outCost, LNKFloat *gradientVector) {
	LNKOptimizationAlgorithmCG *self = tempSelf;
	assert(self);
	assert(inputVector);
	assert(outCost);
	assert(gradientVector);
	
	id<LNKOptimizationAlgorithmDelegate> delegate = self->_delegate;
	LNKRange range = LNKRangeMake(0, self->_exampleCount);
	[delegate optimizationAlgorithmWillBeginIterationWithInputVector:inputVector];
	const LNKFloat cost = [delegate costForOptimizationAlgorithm];
	[delegate computeGradientForOptimizationAlgorithm:gradientVector inRange:range];
	
	*outCost = cost;
}

- (void)runWithParameterVector:(LNKVector)vector exampleCount:(LNKSize)exampleCount delegate:(id<LNKOptimizationAlgorithmDelegate>)delegate {
	NSParameterAssert(vector.data);
	NSParameterAssert(vector.length);
	NSParameterAssert(delegate);
	NSParameterAssert(exampleCount);
	
	_delegate = delegate;
	_exampleCount = exampleCount;
	tempSelf = self;
	
#ifdef DEBUG
	int result = fmincg(_fmincg_evaluate, (LNKFloat *)vector.data, (int)vector.length, (int)_iterationCount);
	NSAssert(result == 0 || result == 1, @"Could not minimize the function");
#else
	fmincg(_fmincg_evaluate, (LNKFloat *)vector.data, (int)vector.length, (int)_iterationCount);
#endif
}

@end
