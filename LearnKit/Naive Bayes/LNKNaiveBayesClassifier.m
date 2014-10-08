//
//  LNKNaiveBayesClassifier.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKNaiveBayesClassifier.h"

#import "_LNKNaiveBayesClassifierAC.h"
#import "LNKDesignMatrix.h"

@implementation LNKNaiveBayesClassifier {
	NSPointerArray *_columnsToValues;
}

- (Class)_classForImplementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm {
#pragma unused(algorithm)
	
	if (implementation == LNKImplementationTypeAccelerate) {
		return [_LNKNaiveBayesClassifierAC class];
	}
	
	NSAssertNotReachable(@"Unsupported implementation type", nil);
	
	return Nil;
}

- (void)registerValues:(NSArray *)values forColumn:(LNKSize)columnIndex {
	NSParameterAssert(values);
	NSParameterAssert(columnIndex < self.designMatrix.columnCount);
	
	if (!_columnsToValues) {
		_columnsToValues = [[NSPointerArray alloc] initWithOptions:NSPointerFunctionsStrongMemory];
		_columnsToValues.count = self.designMatrix.columnCount;
	}
	
	[_columnsToValues insertPointer:values atIndex:columnIndex];
}

- (NSPointerArray *)_columnsToValues {
	return _columnsToValues;
}

- (void)dealloc {
	[_columnsToValues release];
	[super dealloc];
}

@end
