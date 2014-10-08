//
//  LNKTypes.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKTypes.h"

@implementation NSNumber (LNKTypes)

+ (NSNumber *)numberWithLNKFloat:(LNKFloat)value {
#if USE_DOUBLE_PRECISION
	return [self numberWithDouble:value];
#else
	return [self numberWithFloat:value];
#endif
}

+ (NSNumber *)numberWithLNKSize:(LNKSize)value {
	return [self numberWithUnsignedLongLong:value];
}

- (LNKFloat)LNKFloatValue {
#if USE_DOUBLE_PRECISION
	return [self doubleValue];
#else
	return [self floatValue];
#endif
}

- (LNKSize)LNKSizeValue {
	return [self unsignedLongLongValue];
}

@end
