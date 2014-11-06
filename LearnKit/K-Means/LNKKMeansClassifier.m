//
//  LNKKMeansClassifier.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKKMeansClassifier.h"

#import "_LNKKMeansClassifierAC.h"
#import "LNKMatrix.h"
#import "LNKPredictorPrivate.h"

@implementation LNKKMeansClassifier {
	LNKFloat *_clusterCentroids;
}

#define DEFAULT_ITERATION_COUNT 100

+ (NSArray *)supportedImplementationTypes {
	return @[ @(LNKImplementationTypeAccelerate) ];
}

+ (NSArray *)supportedAlgorithms {
	return nil;
}

+ (Class)_classForImplementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(Class)algorithm {
#pragma unused(implementationType)
#pragma unused(algorithm)
	
	return [_LNKKMeansClassifierAC class];
}


- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm classes:(LNKClasses *)classes {
	if (classes.count < 2) {
		@throw [NSException exceptionWithName:NSInternalInconsistencyException reason:@"At least two clusters should be used when running k-means" userInfo:nil];
	}
	
	if (matrix.hasBiasColumn) {
		@throw [NSException exceptionWithName:NSInternalInconsistencyException reason:@"The matrix used with k-means should not have a bias column" userInfo:nil];
	}
	
	if (classes.count >= matrix.exampleCount) {
		@throw [NSException exceptionWithName:NSInternalInconsistencyException reason:@"The number of clusters should be less than the number of examples" userInfo:nil];
	}
	
	self = [super initWithMatrix:matrix implementationType:implementation optimizationAlgorithm:algorithm classes:classes];
	if (self) {
		_iterationCount = DEFAULT_ITERATION_COUNT;
		_clusterCentroids = LNKFloatAlloc(classes.count * matrix.columnCount);
	}
	return self;
}

- (void)dealloc {
	free(_clusterCentroids);
	[super dealloc];
}

- (void)setIterationCount:(LNKSize)iterationCount {
	if (iterationCount == _iterationCount)
		return;
	
	if (iterationCount < 1) {
		@throw [NSException exceptionWithName:NSInternalInconsistencyException reason:@"The iteration count must be >= 1" userInfo:nil];
	}
	
	[self willChangeValueForKey:@"iterationCount"];
	_iterationCount = iterationCount;
	[self didChangeValueForKey:@"iterationCount"];
}

- (LNKFloat *)_clusterCentroids {
	return _clusterCentroids;
}

- (void)_setClusterCentroids:(const LNKFloat *)clusterCentroids {
	NSParameterAssert(clusterCentroids);
	LNKFloatCopy(_clusterCentroids, clusterCentroids, self.classes.count * self.matrix.columnCount);
}

- (const LNKFloat *)centroidForClusterAtIndex:(LNKSize)clusterIndex {
	if (clusterIndex >= self.classes.count)
		[NSException raise:NSGenericException format:@"The cluster index is out-of-bounds"];
		
	return _clusterCentroids + clusterIndex * self.matrix.columnCount;
}

@end
