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

- (LNKFloat)valueWithEpoch:(LNKSize)epoch
{
#pragma unused(epoch)
	return _value;
}

@end

@implementation LNKDecayingAlpha

+ (instancetype)withA:(LNKFloat)a b:(LNKFloat)b {
	return [[[self alloc] initWithA:a b:b] autorelease];
}

- (instancetype)initWithA:(LNKFloat)a b:(LNKFloat)b {
	if (!(self = [super init])) {
		return nil;
	}
	
	_a = a;
	_b = b;
	
	return self;
}

- (LNKFloat)valueWithEpoch:(LNKSize)epoch
{
	return (LNKFloat)1 / (_a + epoch * _b);
}

@end


@implementation LNKOptimizationAlgorithmNormalEquations

- (void)runWithParameterVector:(LNKVector)vector rowCount:(LNKSize)rowCount delegate:(id<LNKOptimizationAlgorithmDelegate>)delegate {
#pragma unused(vector)
#pragma unused(rowCount)
#pragma unused(delegate)
	[NSException raise:NSInternalInconsistencyException format:@"The implementation of normal equations is currently up to the learning algorithm itself"];
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
	
	self = [super init];
	if (self) {
		_alpha = [alpha retain];
		_iterationCount = iterationCount;
		_convergenceThreshold = convergenceThreshold;
	}
	return self;
}

- (void)runWithParameterVector:(LNKVector)vector rowCount:(LNKSize)rowCount delegate:(id<LNKOptimizationAlgorithmDelegate>)delegate {
#pragma unused(vector)
#pragma unused(rowCount)
#pragma unused(delegate)
	[NSException raise:NSInternalInconsistencyException format:@"This method must be overriden by subclasses"];
}

- (void)dealloc {
	[_alpha release];
	[super dealloc];
}

@end

@implementation LNKOptimizationAlgorithmStochasticGradientDescent

- (void)runWithParameterVector:(LNKVector)vector rowCount:(LNKSize)rowCount delegate:(id<LNKOptimizationAlgorithmDelegate>)delegate {
	NSParameterAssert(vector.data);
	NSParameterAssert(vector.length);
	NSParameterAssert(delegate);
	NSParameterAssert(rowCount);

	const LNKSize iterationCount = self.iterationCount;
	const LNKSize batchCount = self.stepCount == NSNotFound ? rowCount : self.stepCount;
	const LNKSize batchSize = rowCount / batchCount;
	
	LNKFloat *weights = LNKFloatAllocAndCopy(vector.data, vector.length);
	
	// Re-used across iterations.
	LNKFloat *gradient = LNKFloatAlloc(vector.length);
	
	for (LNKSize iteration = 0; iteration < iterationCount; iteration++) {
		const LNKFloat alpha = [self.alpha valueWithEpoch:iteration];

		[delegate optimizationAlgorithmWillBeginIteration];
		
		for (NSUInteger batch = 0; batch < batchCount; batch++) {
			LNKRange range = LNKRangeMake(batch * batchSize, batch == batchCount - 1 ? rowCount - batch * batchSize : batchSize);
			
			[delegate optimizationAlgorithmWillBeginWithInputVector:weights];
			[delegate computeGradientForOptimizationAlgorithm:gradient inRange:range];
			
			// Multiply by negative alpha to avoid vsub.
			const LNKFloat negAlpha = -alpha;
			LNK_vsma(gradient, UNIT_STRIDE, &negAlpha, weights, UNIT_STRIDE, weights, UNIT_STRIDE, vector.length);
		}
	}
	
	free(gradient);
	free(weights);
}

@end

@implementation LNKOptimizationAlgorithmLBFGS

- (void)runWithParameterVector:(LNKVector)vector rowCount:(LNKSize)rowCount delegate:(id<LNKOptimizationAlgorithmDelegate>)delegate {
#pragma unused(vector)
#pragma unused(rowCount)
#pragma unused(delegate)
	[NSException raise:NSInternalInconsistencyException format:@"This method must be overriden by subclasses"];
}

@end

@implementation LNKOptimizationAlgorithmCG {
	id<LNKOptimizationAlgorithmDelegate> _delegate;
	LNKSize _rowCount;
}

- (instancetype)init {
	self = [super init];
	if (self) {
		_iterationCount = 100;
	}
	return self;
}

static LNKOptimizationAlgorithmCG *tempSelf = nil;

static void _fmincg_evaluate(LNKFloat *inputVector, LNKFloat *outCost, LNKFloat *gradientVector) {
	LNKOptimizationAlgorithmCG *self = tempSelf;
	NSCAssert(self, @"The self reference must not be nil");
	NSCAssert(inputVector, @"The input vector must not be NULL");
	NSCAssert(outCost, @"The output cost vector must not be NULL");
	NSCAssert(gradientVector, @"The gradient vector must not be NULL");
	
	id<LNKOptimizationAlgorithmDelegate> delegate = self->_delegate;
	LNKRange range = LNKRangeMake(0, self->_rowCount);
	
	[delegate optimizationAlgorithmWillBeginWithInputVector:inputVector];
	const LNKFloat cost = [delegate costForOptimizationAlgorithm];
	[delegate computeGradientForOptimizationAlgorithm:gradientVector inRange:range];
	
	*outCost = cost;
}

- (void)runWithParameterVector:(LNKVector)vector rowCount:(LNKSize)rowCount delegate:(id<LNKOptimizationAlgorithmDelegate>)delegate {
	NSParameterAssert(vector.data);
	NSParameterAssert(vector.length);
	NSParameterAssert(delegate);
	NSParameterAssert(rowCount);
	
	_delegate = delegate;
	_rowCount = rowCount;
	tempSelf = self;
	
#ifdef DEBUG
	int result = fmincg(_fmincg_evaluate, (LNKFloat *)vector.data, (int)vector.length, (int)_iterationCount);
	NSAssert(result == 0 || result == 1, @"Could not minimize the function");
#else
	fmincg(_fmincg_evaluate, (LNKFloat *)vector.data, (int)vector.length, (int)_iterationCount);
#endif
}

@end
