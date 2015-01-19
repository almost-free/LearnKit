//
//  LNKNeuralNetLayer.m
//  LearnKit
//
//  Copyright (c) 2015 Matt Rajca. All rights reserved.
//

#import "LNKNeuralNetLayer.h"

#import "LNKAccelerate.h"

@implementation LNKNeuralNetLayer

- (instancetype)initWithUnitCount:(LNKSize)unitCount {
	self = [super init];
	if (self) {
		_unitCount = unitCount;
	}
	return self;
}

- (instancetype)initWithClasses:(LNKClasses *)classes {
	if (!classes.count)
		[NSException raise:NSInvalidArgumentException format:@"Classes must be specified"];
	
	self = [self initWithUnitCount:classes.count];
	if (self) {
		_classes = [classes retain];
	}
	return self;
}

- (BOOL)isOutputLayer {
	return _classes != nil;
}

- (void)dealloc {
	[_classes release];
	[super dealloc];
}

- (LNKActivationFunction)activationFunction {
	[NSException raise:NSGenericException format:@"%s must be overriden", __PRETTY_FUNCTION__];
	return NULL;
}

@end


@implementation LNKNeuralNetSigmoidLayer

- (LNKActivationFunction)activationFunction {
	return ^(LNKFloat *vector, LNKSize length) {
		LNK_vsigmoid(vector, length);
	};
}

@end
