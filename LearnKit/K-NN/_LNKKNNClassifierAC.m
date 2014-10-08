//
//  _LNKKNNClassifierAC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKKNNClassifierAC.h"

#import "LNKDesignMatrix.h"

typedef struct {
	LNKFloat distance;
	LNKSize index;
} _LNKDistanceBucket;

@implementation _LNKKNNClassifierAC

- (void)train {
	// k-NN does not involve training.
}

- (id)predictValueForFeatureVector:(const LNKFloat *)featureVector length:(LNKSize)length {
	if (!featureVector) {
		@throw [NSException exceptionWithName:NSGenericException reason:@"The feature vector must not be NULL" userInfo:nil];
	}
	
	LNKDesignMatrix *designMatrix = self.designMatrix;
	
	if (length != designMatrix.columnCount) {
		@throw [NSException exceptionWithName:NSGenericException reason:@"The length of the feature vector is incompatible with the design matrix" userInfo:nil];
	}
	
	const LNKSize exampleCount = designMatrix.exampleCount;
	const LNKFloat *outputVector = designMatrix.outputVector;
	const LNKSize k = self.k;
	const LNKKNNDistanceFunction distanceFunction = self.distanceFunction;
	
	_LNKDistanceBucket *closestExamples = calloc(sizeof(_LNKDistanceBucket), k);
	
	// Find the k closest examples.
	for (LNKSize example = 0; example < exampleCount; example++) {
		const LNKFloat *exampleRow = [designMatrix exampleAtIndex:example];
		const LNKFloat distance = distanceFunction(exampleRow, featureVector, length);
		
		if (example < k) {
			closestExamples[example].distance = distance;
			closestExamples[example].index = example;
		}
		else {
			for (LNKSize kOffset = 0; kOffset < k; kOffset++) {
				if (distance < closestExamples[kOffset].distance) {
					closestExamples[kOffset].distance = distance;
					closestExamples[kOffset].index = example;
					break;
				}
			}
		}
	}
	
	// Vote for the item with the most-frequent class.
	NSCountedSet *frequencies = [[NSCountedSet alloc] initWithCapacity:k];
	
	for (LNKSize kOffset = 0; kOffset < k; kOffset++) {
		const _LNKDistanceBucket example = closestExamples[kOffset];
		const LNKFloat outputValue = outputVector[example.index];
		
		LNKClass *class = [[LNKClass classWithUnsignedInteger:outputValue] retain];
		
		[frequencies addObject:class];
		[class release];
	}
	
	free(closestExamples);
	
	LNKClass *bestClass = nil;
	NSUInteger bestFrequency = 0;
	
	for (LNKClass *class in frequencies) {
		const NSUInteger classFrequency = [frequencies countForObject:class];
		
		if (classFrequency > bestFrequency) {
			bestFrequency = classFrequency;
			bestClass = [class retain];
		}
	}
	
	[frequencies release];
	
	return [bestClass autorelease];
}

@end
