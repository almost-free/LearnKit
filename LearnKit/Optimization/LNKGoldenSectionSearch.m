//
//  LNKGoldenSectionSearch.m
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKGoldenSectionSearch.h"

#import "LNKAccelerate.h"

@interface LNKGoldenSectionSearch ()
@property (readwrite) LNKFloat optimalX;
@end

@implementation LNKGoldenSectionSearch {
	LNKUnivariateFunction _function;
	LNKSearchInterval _interval;
	LNKFloat _tolerance;
}

- (instancetype)initWithFunction:(LNKUnivariateFunction)function searchInterval:(LNKSearchInterval)interval
{
	return [self initWithFunction:function searchInterval:interval tolerance:LNK_pow(10, -8)];
}

- (instancetype)initWithFunction:(LNKUnivariateFunction)function searchInterval:(LNKSearchInterval)interval tolerance:(LNKFloat)tolerance
{
	if (function == nil) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"A function must be specified" userInfo:nil];
	}

	if (interval.end < interval.start) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"The end location must not precede the start location" userInfo:nil];
	}

	if (!(self = [super init])) {
		return nil;
	}

	_function = [function copy];
	_interval = interval;
	_tolerance = tolerance;
	_optimalX = LNKFloatMax;

	return self;
}

- (void)dealloc
{
	[_function release];
	[super dealloc];
}

- (void)main
{
	const LNKFloat goldenSection = (LNK_sqrt(5) - 1) / 2;

	LNKFloat currentStart = _interval.start;
	LNKFloat currentEnd = _interval.end;

	while (LNK_fabs(currentEnd - currentStart) > _tolerance) {
		const LNKFloat length = currentEnd - currentStart;
		const LNKFloat m1 = currentStart + (1 - goldenSection) * length;
		const LNKFloat m2 = currentStart + goldenSection * length;

		if (_function(m1) < _function(m2)) {
			currentEnd = m2;
		}
		else {
			currentStart = m1;
		}
	}

	self.optimalX = (currentStart + currentEnd) / 2;
}

@end
