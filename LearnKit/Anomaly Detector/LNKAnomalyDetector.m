//
//  LNKAnomalyDetector.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKAnomalyDetector.h"

#import "_LNKAnomalyDetectorAC.h"
#import "LNKAccelerate.h"
#import "LNKMatrix.h"

@interface LNKAnomalyDetector (Private)

@property (nonatomic) LNKFloat *_muVector;
@property (nonatomic) LNKFloat *_sigmaMatrix;

- (LNKFloat)_probabilityWithFeatureVector:(const LNKFloat *)featureVector length:(LNKSize)length;

@end


@implementation LNKAnomalyDetector

#define DEFAULT_THRESHOLD 0.01

+ (NSArray *)supportedImplementationTypes {
	return @[ @(LNKImplementationTypeAccelerate) ];
}

+ (NSArray *)supportedAlgorithms {
	return nil;
}

+ (Class)_classForImplementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(Class)algorithm {
#pragma unused(implementationType)
#pragma unused(algorithm)
	
	return [_LNKAnomalyDetectorAC class];
}


- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm classes:(LNKClasses *)classes {
#pragma unused(classes)
	
	if (matrix.hasBiasColumn) {
		@throw [NSException exceptionWithName:NSGenericException reason:@"The matrix used for anomaly detection should not have a bias column" userInfo:nil];
	}
	
	if (matrix.exampleCount < matrix.columnCount) {
		@throw [NSException exceptionWithName:NSGenericException reason:@"The matrix used for anomaly detection must have more example rows than columns" userInfo:nil];
	}
	
	self = [super initWithMatrix:matrix implementationType:implementation optimizationAlgorithm:algorithm classes:[LNKClasses withCount:2]];
	if (self) {
		_threshold = DEFAULT_THRESHOLD;
	}
	return self;
}

- (LNKFloat *)_muVector {
	NSAssertNotReachable(@"Subclasses should override %s", __PRETTY_FUNCTION__);
	return NULL;
}

- (LNKFloat *)_sigmaMatrix {
	NSAssertNotReachable(@"Subclasses should override %s", __PRETTY_FUNCTION__);
	return NULL;
}

- (void)_setMuVector:(LNKFloat *)vector {
#pragma unused(vector)
	NSAssertNotReachable(@"Subclasses should override %s", __PRETTY_FUNCTION__);
}

- (void)_setSigmaMatrix:(LNKFloat *)matrix {
#pragma unused(matrix)
	NSAssertNotReachable(@"Subclasses should override %s", __PRETTY_FUNCTION__);
}

@end


LNKFloat LNKFindAnomalyThreshold(LNKMatrix *matrix, LNKMatrix *cvMatrix) {
	const LNKSize cvColumnCount = cvMatrix.columnCount;
	
	if (matrix.columnCount != cvColumnCount) {
		@throw [NSException exceptionWithName:NSGenericException reason:@"The cross validation matrix must have the same number of columns as the matrix" userInfo:nil];
	}
	
	const LNKSize cvExampleCount = cvMatrix.exampleCount;
	
	LNKAnomalyDetector *detector = [[LNKAnomalyDetector alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:nil classes:nil];
	[detector train];
	
	LNKAnomalyDetector *cvDetector = [[LNKAnomalyDetector alloc] initWithMatrix:cvMatrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:nil classes:nil];
	[cvDetector _setMuVector:[detector _muVector]];
	[cvDetector _setSigmaMatrix:[detector _sigmaMatrix]];
	
	LNKFloat *pValues = LNKFloatAlloc(cvExampleCount);
	
	for (LNKSize example = 0; example < cvExampleCount; example++) {
		pValues[example] = [cvDetector _probabilityWithFeatureVector:[cvMatrix exampleAtIndex:example] length:cvColumnCount];
	}
	
	LNKFloat max, min;
	LNK_maxv(pValues, UNIT_STRIDE, &max, cvExampleCount);
	LNK_minv(pValues, UNIT_STRIDE, &min, cvExampleCount);
	
	const LNKFloat stepSize = (max - min) / 1000;
	const LNKFloat *outputVector = cvMatrix.outputVector;
	
	LNKFloat bestEpsilon = 0;
	LNKFloat bestF1 = 0;
	
	for (LNKFloat epsilon = min; epsilon < max; epsilon += stepSize) {
		LNKSize truePositives = 0;
		LNKSize falsePositives = 0;
		LNKSize falseNegatives = 0;
		
		for (LNKSize example = 0; example < cvExampleCount; example++) {
			BOOL anomaly = pValues[example] < epsilon;
			
			if (anomaly) {
				if (outputVector[example])
					truePositives++;
				else
					falsePositives++;
			}
			else if (outputVector[example])
				falseNegatives++;
		}
		
		const LNKFloat precision = (LNKFloat)truePositives / (truePositives + falsePositives);
		const LNKFloat recall = (LNKFloat)truePositives / (truePositives + falseNegatives);
		
		const LNKFloat f1 = (2 * precision * recall) / (precision * recall);
		
		if (f1 > bestF1) {
			bestF1 = f1;
			bestEpsilon = epsilon;
		}
	}
	
	free(pValues);
	
	[detector release];
	[cvDetector release];
	
	return bestEpsilon;
}
