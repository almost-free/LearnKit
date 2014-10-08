//
//  _LNKOneVsAllLogRegClassifierLBFGS_AC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKOneVsAllLogRegClassifierLBFGS_AC.h"

#import "_LNKLogRegClassifierLBFGS_AC.h"
#import "LNKClassifierPrivate.h"
#import "LNKDesignMatrix.h"
#import "LNKPredictorPrivate.h"

@implementation _LNKOneVsAllLogRegClassifierLBFGS_AC {
	NSMapTable *_classesToClassifiers;
}

- (void)train {
	NSAssert(self.classes.count >= 2, @"There should be at least two output classes");
	
	if (!_classesToClassifiers)
		_classesToClassifiers = [[NSMapTable strongToStrongObjectsMapTable] retain];
	
	for (LNKClass *class in self.classes) {
		LNKDesignMatrix *designMatrixCopy = [self.designMatrix copy];
		[designMatrixCopy modifyOutputVector:^(LNKFloat *outputVector, LNKSize m) {
			for (LNKSize i = 0; i < m; i++) {
				outputVector[i] = (outputVector[i] == class.unsignedIntegerValue) ? 1 : 0;
			}
		}];
		
		_LNKLogRegClassifierLBFGS_AC *classifier = [[_LNKLogRegClassifierLBFGS_AC alloc] initWithDesignMatrix:designMatrixCopy implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:self.algorithm];
		[designMatrixCopy release];
		[classifier train];
		
		[_classesToClassifiers setObject:classifier forKey:class];
		[classifier release];
	}
}

- (void)_predictValueForFeatureVector:(const LNKFloat *)featureVector length:(LNKSize)length {
	NSParameterAssert(featureVector);
	NSParameterAssert(length);
	
	for (LNKClass *class in _classesToClassifiers) {
		LNKLogRegClassifier *classifier = [_classesToClassifiers objectForKey:class];
		LNKFloat probability = [[classifier predictValueForFeatureVector:featureVector length:length] LNKFloatValue];
		
		[self _didPredictProbability:probability forClass:class];
	}
}

- (void)dealloc {
	[_classesToClassifiers release];
	[super dealloc];
}

@end
