//
//  LNKClasses.m
//  LearnKit
//
//  Copyright (c) 2015 Matt Rajca. All rights reserved.
//

#import "LNKClasses.h"

@implementation LNKClass

+ (instancetype)classWithUnsignedInteger:(NSUInteger)value {
	return [[[self alloc] initWithUnsignedInteger:value] autorelease];
}

- (instancetype)initWithUnsignedInteger:(NSUInteger)value {
	if (!(self = [super init]))
		return nil;
	
	_unsignedIntegerValue = value;
	
	return self;
}

- (NSUInteger)hash {
	return _unsignedIntegerValue;
}

- (BOOL)isEqual:(LNKClass *)object {
	return [object isKindOfClass:[LNKClass class]] && _unsignedIntegerValue == object->_unsignedIntegerValue;
}

@end


@implementation LNKClasses {
	NSArray<LNKClass *> *_classes;
	LNKClassMapper _mapper;
}

static NSArray<LNKClass *> *_LNKIntegersInRange(NSRange range) {
	NSMutableArray<LNKClass *> *array = [[NSMutableArray alloc] init];
	
	for (NSUInteger n = range.location; n < NSMaxRange(range); n++) {
		[array addObject:[LNKClass classWithUnsignedInteger:n]];
	}
	
	return [array autorelease];
}

+ (instancetype)withRange:(NSRange)range {
	NSParameterAssert(range.length);
	NSArray<LNKClass *> *classes = _LNKIntegersInRange(range);
	
	return [self withClasses:classes mapper:^NSUInteger(LNKClass *aClass) {
		return [aClass unsignedIntegerValue] - range.location;
	}];
}

+ (instancetype)withCount:(LNKSize)count {
	NSParameterAssert(count);
	return [self withRange:NSMakeRange(0, count)];
}

+ (instancetype)withClasses:(NSArray<LNKClass *> *)classes mapper:(LNKClassMapper)mapper {
	return [[self alloc] initWithClasses:classes mapper:mapper];
}

- (instancetype)initWithClasses:(NSArray<LNKClass *> *)classes mapper:(LNKClassMapper)mapper {
	NSParameterAssert(classes.count);
	NSParameterAssert(mapper);
	
	if (!(self = [super init]))
		return nil;
	
	_classes = [classes retain];
	_mapper = [mapper copy];
	
	return self;
}

- (NSUInteger)indexForClass:(LNKClass *)aClass {
	NSParameterAssert(aClass);
	return _mapper(aClass);
}

- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(__unsafe_unretained id [])buffer count:(NSUInteger)len {
	return [_classes countByEnumeratingWithState:state objects:buffer count:len];
}

- (NSUInteger)count {
	return _classes.count;
}

- (void)dealloc {
	[_classes release];
	[_mapper release];
	[super dealloc];
}

@end
