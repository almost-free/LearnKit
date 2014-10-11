//
//  LNKKNNClassifier.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKKNNClassifier.h"

#import "_LNKKNNClassifierAC.h"
#import "LNKAccelerate.h"
#import "LNKMatrix.h"

const LNKKNNDistanceFunction LNKKNNEuclideanDistanceFunction = ^LNKFloat(const LNKFloat *example1, const LNKFloat *example2, LNKSize n) {
	LNKFloat result;
	LNKVectorDistance(example1, example2, &result, n);
	
	return result;
};

@implementation LNKKNNClassifier

#define DEFAULT_K 1

- (Class)_classForImplementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm {
#pragma unused(algorithm)
	
	if (implementation == LNKImplementationTypeAccelerate) {
		return [_LNKKNNClassifierAC class];
	}
	
	NSAssertNotReachable(@"Unsupported implementation type", nil);
	
	return Nil;
}

- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm classes:(LNKClasses *)classes {
	if (classes.count < 2) {
		@throw [NSException exceptionWithName:NSGenericException reason:@"At least two classes should be specified when running k-NN" userInfo:nil];
	}
	
	if (matrix.hasBiasColumn) {
		@throw [NSException exceptionWithName:NSGenericException reason:@"The matrix used with k-NN should not have a bias column" userInfo:nil];
	}
	
	self = [super initWithMatrix:matrix implementationType:implementation optimizationAlgorithm:algorithm classes:classes];
	if (self) {
		_k = DEFAULT_K;
		_distanceFunction = [LNKKNNEuclideanDistanceFunction copy];
		_outputFunction = LNKKNNOutputFunctionMostFrequent;
	}
	return self;
}

- (void)setK:(LNKSize)k {
	if (k < 1) {
		@throw [NSException exceptionWithName:NSGenericException reason:@"The parameter k must not be less than 1" userInfo:nil];
	}
	
	if (k >= self.matrix.exampleCount) {
		@throw [NSException exceptionWithName:NSGenericException reason:@"The parameter k must be less than the number of examples" userInfo:nil];
	}
	
	if (_k != k) {
		[self willChangeValueForKey:@"k"];
		_k = k;
		[self didChangeValueForKey:@"k"];
	}
}

- (void)setDistanceFunction:(LNKKNNDistanceFunction)distanceFunction {
	if (distanceFunction == nil) {
		@throw [NSException exceptionWithName:NSGenericException reason:@"The distance function must be specified" userInfo:nil];
	}
	
	if (_distanceFunction == distanceFunction)
		return;
	
	[self willChangeValueForKey:@"distanceFunction"];
	
	[_distanceFunction release];
	_distanceFunction = [distanceFunction retain];
	
	[self didChangeValueForKey:@"distanceFunction"];
}

- (void)dealloc {
	[_distanceFunction release];
	[super dealloc];
}

@end
