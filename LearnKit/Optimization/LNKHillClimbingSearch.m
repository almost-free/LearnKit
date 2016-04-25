//
//  LNKHillClimbingSearch.m
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKHillClimbingSearch.h"

#import "LNKAccelerate.h"

@interface LNKHillClimbingSearch ()
@property LNKFloat optimalY;
@end

@implementation LNKHillClimbingSearch {
	LNKMultivariateFunction _function;
	LNKSize _parameterCount;
	LNKFloat *_parameters;
	LNKFloat *_stepSize;
	LNKFloat *_parameterMins;
	LNKFloat *_parameterMaxs;
}

- (instancetype)initWithFunction:(LNKMultivariateFunction)function parameters:(LNKVector)parameters stepSizes:(LNKVector)stepSizes minParameter:(LNKVector)minParameters maxParameters:(LNKVector)maxParameters
{
	if (function == nil) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"The function must not be nil" userInfo:nil];
	}

	if (parameters.length == 0) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"At least one parameter must be specified" userInfo:nil];
	}

	if (parameters.length != stepSizes.length) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"The number of parameters should be equal to number of step sizes" userInfo:nil];
	}

	if (parameters.length != minParameters.length) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"The number of parameters should be equal to number of min values" userInfo:nil];
	}

	if (parameters.length != maxParameters.length) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"The number of parameters should be equal to number of max values" userInfo:nil];
	}

	if (!(self = [super init])) {
		return nil;
	}

	_function = [function copy];
	_parameterCount = parameters.length;
	_parameters = LNKFloatAllocAndCopy(parameters.data, _parameterCount);
	_stepSize = LNKFloatAllocAndCopy(stepSizes.data, _parameterCount);
	_parameterMins = LNKFloatAllocAndCopy(minParameters.data, _parameterCount);
	_parameterMaxs = LNKFloatAllocAndCopy(maxParameters.data, _parameterCount);

	return self;
}

- (void)dealloc
{
	[_function release];
	free(_parameters);
	free(_stepSize);
	free(_parameterMins);
	free(_parameterMaxs);
	[super dealloc];
}

- (void)main
{
	LNKFloat bestValue = _function(LNKVectorWrapUnsafe(_parameters, _parameterCount));

	while (YES) {
		LNKFloat maxY = bestValue;
		LNKSize maxIndex = 0;
		LNKFloat maxParameter = 0;

		for (LNKSize i = 0; i < _parameterCount; i++) {
			const LNKFloat savedParameter = _parameters[i];

			_parameters[i] = savedParameter + _stepSize[i];
			if (_parameters[i] <= _parameterMaxs[i]) {
				const LNKFloat y = _function(LNKVectorWrapUnsafe(_parameters, _parameterCount));
				if (y > maxY) {
					maxY = y;
					maxIndex = i;
					maxParameter = _parameters[i];
				}
			}

			_parameters[i] = savedParameter - _stepSize[i];
			if (_parameters[i] >= _parameterMins[i]) {
				const LNKFloat y = _function(LNKVectorWrapUnsafe(_parameters, _parameterCount));
				if (y > maxY) {
					maxY = y;
					maxIndex = i;
					maxParameter = _parameters[i];
				}
			}

			_parameters[i] = savedParameter;
		}

		if (maxY <= bestValue) {
			self.optimalY = bestValue;
			break;
		}

		_parameters[maxIndex] = maxParameter;
		bestValue = maxY;
	}
}

@end
