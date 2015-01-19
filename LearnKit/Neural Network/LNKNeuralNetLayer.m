//
//  LNKNeuralNetLayer.m
//  LearnKit
//
//  Copyright (c) 2015 Matt Rajca. All rights reserved.
//

#import "LNKNeuralNetLayer.h"

@implementation LNKNeuralNetLayer

- (instancetype)initWithUnitCount:(LNKSize)unitCount {
	self = [super init];
	if (self) {
		_unitCount = unitCount;
	}
	return self;
}

@end


@implementation LNKNeuralNetOutputLayer

- (instancetype)initWithClasses:(LNKClasses *)classes {
	if (!classes.count)
		[NSException raise:NSInvalidArgumentException format:@"Classes must be specified"];
	
	self = [super initWithUnitCount:classes.count];
	if (self) {
		_classes = [classes retain];
	}
	return self;
}

- (void)dealloc {
	[_classes release];
	[super dealloc];
}

@end
