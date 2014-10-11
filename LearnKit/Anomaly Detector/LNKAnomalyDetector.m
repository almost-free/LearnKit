//
//  LNKAnomalyDetector.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKAnomalyDetector.h"

#import "_LNKAnomalyDetectorAC.h"
#import "LNKMatrix.h"

@interface LNKAnomalyDetector (Private)

@property (nonatomic) LNKFloat *_muVector;
@property (nonatomic) LNKFloat *_sigmaMatrix;

- (LNKFloat)_probabilityWithFeatureVector:(const LNKFloat *)featureVector length:(LNKSize)length;

@end


@implementation LNKAnomalyDetector

#define DEFAULT_THRESHOLD 0.01

- (Class)_classForImplementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm {
#pragma unused(algorithm)
	
	if (implementation == LNKImplementationTypeAccelerate) {
		return [_LNKAnomalyDetectorAC class];
	}
	
	NSAssertNotReachable(@"Unsupported implementation type", nil);
	
	return Nil;
}

- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm classes:(LNKClasses *)classes {
#pragma unused(classes)
	
	if (matrix.hasBiasColumn) {
		@throw [NSException exceptionWithName:NSGenericException reason:@"The matrix used for anomaly detection should not have a bias column" userInfo:nil];
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
	
	free(pValues);
	
	[detector release];
	[cvDetector release];
	
	return 0;
}
