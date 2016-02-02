//
//  LNKNaiveBayesClassifier.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKNaiveBayesClassifier.h"

#import "_LNKNaiveBayesClassifierAC.h"
#import "LNKMatrix.h"

@implementation LNKNaiveBayesClassifier {
	NSPointerArray *_columnsToValues;
}

+ (NSArray<NSNumber *> *)supportedImplementationTypes {
	return @[ @(LNKImplementationTypeAccelerate) ];
}

+ (NSArray<Class> *)supportedAlgorithms {
	return nil;
}

+ (Class)_classForImplementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(Class)algorithm {
#pragma unused(implementationType)
#pragma unused(algorithm)
	
	return [_LNKNaiveBayesClassifierAC class];
}

- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm classes:(LNKClasses *)classes {
	self = [super initWithMatrix:matrix implementationType:implementation optimizationAlgorithm:algorithm classes:classes];
	if (self) {
		_performsLaplacianSmoothing = YES;
		_laplacianSmoothingFactor = 1;
	}
	return self;
}

- (void)registerValues:(NSArray<NSNumber *> *)values forColumn:(LNKSize)columnIndex {
	if (!values)
		[NSException raise:NSGenericException format:@"The array of values must not be nil"];
	
	const LNKSize columnCount = self.matrix.columnCount;
	
	if (columnIndex >= columnCount)
		[NSException raise:NSGenericException format:@"The given index (%lld) is out-of-bounds (%lld)", columnIndex, columnCount];
	
	if (!_columnsToValues) {
		_columnsToValues = [[NSPointerArray alloc] initWithOptions:NSPointerFunctionsStrongMemory];
		_columnsToValues.count = columnCount;
	}
	
	[_columnsToValues insertPointer:values atIndex:columnIndex];
}

- (NSPointerArray *)_columnsToValues {
	return _columnsToValues;
}

- (id)predictValueForFeatureVector:(LNKVector)featureVector probability:(LNKFloat *)outProbability {
#pragma unused(featureVector)
#pragma unused(outProbability)

	[NSException raise:NSInternalInconsistencyException format:@"%s must be overriden by subclasses", __PRETTY_FUNCTION__];
	return nil;
}

- (void)dealloc {
	[_columnsToValues release];
	[super dealloc];
}

@end
