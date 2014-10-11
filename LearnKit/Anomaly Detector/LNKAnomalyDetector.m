//
//  LNKAnomalyDetector.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKAnomalyDetector.h"

#import "_LNKAnomalyDetectorAC.h"
#import "LNKMatrix.h"

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

@end
