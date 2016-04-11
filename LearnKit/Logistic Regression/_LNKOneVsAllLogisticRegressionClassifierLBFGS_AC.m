//
//  _LNKOneVsAllLogisticRegressionClassifierLBFGS_AC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKOneVsAllLogisticRegressionClassifierLBFGS_AC.h"

#import "_LNKLogisticRegressionClassifierLBFGS_AC.h"
#import "LNKClassifierPrivate.h"
#import "LNKMatrix.h"
#import "LNKPredictorPrivate.h"

@implementation _LNKOneVsAllLogisticRegressionClassifierLBFGS_AC {
	NSMapTable *_classesToClassifiers;
}

- (void)train {
	NSAssert(self.classes.count >= 2, @"There should be at least two output classes");
	
	if (!_classesToClassifiers)
		_classesToClassifiers = [[NSMapTable strongToStrongObjectsMapTable] retain];
	
	for (LNKClass *class in self.classes) {
		LNKMatrix *matrixCopy = [self.matrix copy];
		[matrixCopy modifyOutputVector:^(LNKFloat *outputVector, LNKSize m) {
			for (LNKSize i = 0; i < m; i++) {
				outputVector[i] = (outputVector[i] == class.unsignedIntegerValue) ? 1 : 0;
			}
		}];
		
		LNKLogisticRegressionClassifier *classifier = [[LNKLogisticRegressionClassifier alloc] initWithMatrix:matrixCopy implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:self.algorithm];
		[matrixCopy release];
		[classifier train];
		
		[_classesToClassifiers setObject:classifier forKey:class];
		[classifier release];
	}
}

- (void)_predictValueForFeatureVector:(LNKVector)featureVector {
	NSParameterAssert(featureVector.data);
	NSParameterAssert(featureVector.length);

	if (featureVector.length != self.matrix.columnCount) {
		[NSException raise:NSGenericException format:@"The length of the feature vector must be equal to the number of columns in the matrix"]; // otherwise, we can't do matrix multiplication
	}
	
	for (LNKClass *class in _classesToClassifiers) {
		LNKLogisticRegressionClassifier *classifier = [_classesToClassifiers objectForKey:class];
		LNKFloat probability = [[classifier predictValueForFeatureVector:featureVector] LNKFloatValue];
		
		[self _didPredictProbability:probability forClass:class];
	}
}

- (void)dealloc {
	[_classesToClassifiers release];
	[super dealloc];
}

@end
