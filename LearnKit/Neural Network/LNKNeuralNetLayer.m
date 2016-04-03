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

- (LNKActivationGradientFunction)activationGradientFunction {
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

- (LNKActivationGradientFunction)activationGradientFunction {
	return ^(const LNKFloat *vector, LNKFloat *outVector, LNKSize length) {
		LNK_vsigmoidgrad(vector, outVector, length);
	};
}

@end


@implementation LNKNeuralNetReLULayer

- (LNKActivationFunction)activationFunction {
	return ^(LNKFloat *vector, LNKSize length) {
		for (LNKSize i = 0; i < length; i++) {
			vector[i] = vector[i] < 0 ? 0 : vector[i];
		}
	};
}

- (LNKActivationGradientFunction)activationGradientFunction {
	return ^(const LNKFloat *vector, LNKFloat *outVector, LNKSize length) {
		for (LNKSize i = 0; i < length; i++) {
			outVector[i] = vector[i] > 0 ? 1 : 0;
		}
	};
}

@end


@implementation LNKNeuralNetTanhLayer

- (LNKActivationFunction)activationFunction {
	return ^(LNKFloat *vector, LNKSize length) {
		const int lengthInt = (int)length;
		LNK_vtanh(vector, vector, &lengthInt);
	};
}

// 1 - tanh^2(x)
- (LNKActivationGradientFunction)activationGradientFunction {
	return ^(const LNKFloat *vector, LNKFloat *outVector, LNKSize length) {
		const int lengthInt = (int)length;
		LNK_vtanh(outVector, vector, &lengthInt);
		const LNKFloat power = 2;
		LNK_vpows(outVector, &power, outVector, &lengthInt);
		LNK_vneg(outVector, UNIT_STRIDE, outVector, UNIT_STRIDE, lengthInt);
		const LNKFloat one = 1;
		LNK_vsadd(outVector, UNIT_STRIDE, &one, outVector, UNIT_STRIDE, lengthInt);
	};
}

@end
