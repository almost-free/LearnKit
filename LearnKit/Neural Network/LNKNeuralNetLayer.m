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
