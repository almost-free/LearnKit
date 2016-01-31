//
//  LNKConfusionMatrix.m
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKConfusionMatrix.h"

#import "LNKClasses.h"
#import "LNKConfusionMatrixPrivate.h"

@implementation LNKConfusionMatrix {
	NSMutableDictionary<LNKClass *, NSCountedSet<LNKClass *> *> *_frequencyTable;
}

- (NSUInteger)frequencyForTrueClass:(LNKClass *)trueClass predictedClass:(LNKClass *)predictedClass {
	if (!trueClass) {
		[NSException raise:NSInvalidArgumentException format:@"The true class must be specified"];
	}

	if (!predictedClass) {
		[NSException raise:NSInvalidArgumentException format:@"The predicted class must be specified"];
	}

	NSCountedSet<LNKClass *> *const trueCounts = _frequencyTable[trueClass];

	if (!trueCounts) {
		return 0;
	}

	return [trueCounts countForObject:predictedClass];
}

- (void)_incrementFrequencyForTrueClass:(LNKClass *)trueClass predictedClass:(LNKClass *)predictedClass {
	if (!trueClass) {
		[NSException raise:NSInvalidArgumentException format:@"The true class must be specified"];
	}

	if (!predictedClass) {
		[NSException raise:NSInvalidArgumentException format:@"The predicted class must be specified"];
	}

	if (_frequencyTable == nil) {
		_frequencyTable = [[NSMutableDictionary alloc] init];
	}

	NSCountedSet<LNKClass *> *trueCounts = _frequencyTable[trueClass];

	if (!trueCounts) {
		trueCounts = [[[NSCountedSet alloc] init] autorelease];
		_frequencyTable[trueClass] = trueCounts;
	}

	[trueCounts addObject:predictedClass];
}

- (void)dealloc {
	[_frequencyTable release];
	[super dealloc];
}

@end
