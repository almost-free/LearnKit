//
//  NSCountedSetAdditions.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "NSCountedSetAdditions.h"

@implementation NSCountedSet (Additions)

- (id)mostFrequentObject {
	__block id bestObject = self.anyObject;
	__block NSUInteger bestFrequency = [self countForObject:bestObject];
	
	[self enumerateObjectsUsingBlock:^(id object, BOOL *stop) {
#pragma unused(stop)
		
		const NSUInteger frequency = [self countForObject:object];
		
		if (frequency > bestFrequency) {
			bestFrequency = frequency;
			bestObject = object;
		}
	}];
	
	return bestObject;
}

@end
