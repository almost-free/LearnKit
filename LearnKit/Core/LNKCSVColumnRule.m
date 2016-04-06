//
//  LNKCSVColumnRule.m
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKCSVColumnRule.h"

@implementation LNKCSVColumnRule {
	id _object;
}

+ (instancetype)deleteRule {
	return [[[self alloc] _initWithType:LNKCSVColumnRuleTypeDelete] autorelease];
}

+ (instancetype)conversionRuleWithBlock:(LNKCSVColumnRuleTypeConversionHandler)block {
	if (block == nil) {
		[NSException raise:NSInvalidArgumentException format:@"The conversion handler block must not be nil"];
	}

	return [[[self alloc] _initWithType:LNKCSVColumnRuleTypeConversion object:[[block copy] autorelease]] autorelease];
}

+ (instancetype)outputRule {
	return [[[self alloc] _initWithType:LNKCSVColumnRuleTypeOutput] autorelease];
}

- (instancetype)_initWithType:(LNKCSVColumnRuleType)type {
	return [self _initWithType:type object:nil];
}

- (instancetype)_initWithType:(LNKCSVColumnRuleType)type object:(id)object {
	if (!(self = [super init])) {
		return nil;
	}

	_type = type;
	_object = [object retain];

	return self;
}

- (void)dealloc {
	[_object release];
	[super dealloc];
}

@end
