//
//  NSIndexSetAdditions.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "NSIndexSetAdditions.h"

@implementation NSIndexSet (Additions)

+ (NSIndexSet *)withCount:(NSUInteger)count {
	return [[[NSIndexSet alloc] initWithIndexesInRange:NSMakeRange(0, count)] autorelease];
}

- (NSIndexSet *)indexSetByRemovingIndex:(NSUInteger)index {
	if (![self containsIndex:index])
		return self;
	
	NSMutableIndexSet *mutableSelf = [self mutableCopy];
	[mutableSelf removeIndex:index];
	
	NSIndexSet *immutableCopy = [mutableSelf copy];
	[mutableSelf release];
	
	return [immutableCopy autorelease];
}

- (NSIndexSet *)indexSetByRandomlySamplingTo:(NSUInteger)size {
	const NSUInteger count = self.count;
	if (size > count) {
		return self;
	}

	NSUInteger indices[count];
	NSRange range = NSMakeRange(0, count);
	[self getIndexes:indices maxCount:count inIndexRange:&range];

	for (NSUInteger i = 0; i < size; i++) {
		const NSUInteger randomIndex = (NSUInteger)arc4random_uniform((uint32_t)count);
		const NSUInteger value = indices[i];
		indices[i] = indices[randomIndex];
		indices[randomIndex] = value;
	}

	NSMutableIndexSet *const randomSet = [[NSMutableIndexSet alloc] init];
	for (NSUInteger i = 0; i < size; i++) {
		[randomSet addIndex:indices[i]];
	}

	NSIndexSet *const randomSetImmutable = [randomSet copy];
	[randomSet release];

	return [randomSetImmutable autorelease];
}

- (void)enumerateAllIndicesUsingBlock:(NSIndexSetSimpleEnumerator)block {
	NSParameterAssert(block);
	
	[self enumerateIndexesUsingBlock:^(NSUInteger index, BOOL *stop) {
#pragma unused(stop)
		block(index);
	}];
}

@end
