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

+ (NSArray *)supportedImplementationTypes {
	return @[ @(LNKImplementationTypeAccelerate) ];
}

+ (NSArray *)supportedAlgorithms {
	return nil;
}

+ (Class)_classForImplementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(Class)algorithm {
#pragma unused(implementationType)
#pragma unused(algorithm)
	
	return [_LNKNaiveBayesClassifierAC class];
}


- (void)registerValues:(NSArray *)values forColumn:(LNKSize)columnIndex {
	NSParameterAssert(values);
	NSParameterAssert(columnIndex < self.matrix.columnCount);
	
	if (!_columnsToValues) {
		_columnsToValues = [[NSPointerArray alloc] initWithOptions:NSPointerFunctionsStrongMemory];
		_columnsToValues.count = self.matrix.columnCount;
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
