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

const LNKKNNDistanceFunction LNKKNNEuclideanDistanceFunction = ^LNKFloat(LNKVector example1, LNKVector example2) {
	LNKFloat result;
	LNKVectorDistance(example1.data, example2.data, &result, example1.length);
	
	return result;
};

@implementation LNKKNNClassifier

#define DEFAULT_K 1

+ (NSArray<NSNumber *> *)supportedImplementationTypes {
	return @[ @(LNKImplementationTypeAccelerate) ];
}

+ (NSArray<Class> *)supportedAlgorithms {
	return @[];
}

+ (Class)_classForImplementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(Class)algorithm {
#pragma unused(implementationType)
#pragma unused(algorithm)
	
	return [_LNKKNNClassifierAC class];
}


- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm classes:(LNKClasses *)classes {
	if (classes.count < 2) {
		@throw [NSException exceptionWithName:NSGenericException reason:@"At least two classes should be specified when running k-NN" userInfo:nil];
	}
	
	if (matrix.hasBiasColumn) {
		@throw [NSException exceptionWithName:NSGenericException reason:@"The matrix used with k-NN should not have a bias column" userInfo:nil];
	}

	LNKMatrix *const normalizedCopy = [matrix.normalizedMatrix retain];

	if (!(self = [super initWithMatrix:normalizedCopy implementationType:implementation optimizationAlgorithm:algorithm classes:classes])) {
		[normalizedCopy release];
		return nil;
	}

	[normalizedCopy release];

	_k = DEFAULT_K;
	_distanceFunction = [LNKKNNEuclideanDistanceFunction copy];
	_outputFunction = LNKKNNOutputFunctionMostFrequent;

	return self;
}

- (void)setK:(LNKSize)k {
	if (k < 1) {
		@throw [NSException exceptionWithName:NSGenericException reason:@"The parameter k must not be less than 1" userInfo:nil];
	}
	
	if (k >= self.matrix.rowCount) {
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
	_distanceFunction = [distanceFunction copy];
	
	[self didChangeValueForKey:@"distanceFunction"];
}

- (void)dealloc {
	[_distanceFunction release];
	[super dealloc];
}

@end
