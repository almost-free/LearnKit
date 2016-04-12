//
//  LNKRegularizationConfiguration.m
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKRegularizationConfiguration.h"

@implementation LNKRegularizationConfiguration

- (instancetype)initWithLambda:(LNKFloat)lambda
{
	if (lambda <= 0) {
		[NSException raise:NSInvalidArgumentException format:@"The lambda value must be greater than 0"];
	}

	if (!(self = [super init])) {
		return nil;
	}

	_lambda = lambda;

	return self;
}

+ (instancetype)withLambda:(LNKFloat)lambda
{
	return [[[self alloc] initWithLambda:lambda] autorelease];
}

@end
