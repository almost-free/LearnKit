//
//  LNKDiscreteProbabilityDistribution.m
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKDiscreteProbabilityDistribution.h"

#import "LNKClasses.h"
#import "LNKClassProbabilityDistributionPrivate.h"
#import "LNKMatrix.h"

@implementation LNKDiscreteProbabilityDistribution {
	NSPointerArray *_columnsToValues;
	LNKFloat **_featureProbabilities;
}

- (instancetype)initWithClasses:(LNKClasses *)classes featureCount:(LNKSize)featureCount {
	if (!(self = [super initWithClasses:classes featureCount:featureCount]))
		return nil;

	_performsLaplacianSmoothing = YES;
	_laplacianSmoothingFactor = 1;

	return self;
}

- (void)_freeBuffers {
	if (_featureProbabilities == NULL) {
		return;
	}

	const LNKSize entryCount = self.featureCount * self.classes.count;

	for (LNKSize i = 0; i < entryCount; i++) {
		if (_featureProbabilities[i] != NULL) {
			free(_featureProbabilities[i]);
		}
	}

	free(_featureProbabilities);
}

- (void)dealloc {
	[_columnsToValues release];
	[self _freeBuffers];
	[super dealloc];
}

- (void)registerValues:(NSArray<NSNumber *> *)values forColumnAtIndex:(LNKSize)columnIndex {
	if (values == nil) {
		[NSException raise:NSGenericException format:@"The array of values must not be nil"];
	}

	const LNKSize columnCount = self.featureCount;

	if (columnIndex >= columnCount) {
		[NSException raise:NSGenericException format:@"The given index (%lld) is out-of-bounds (%lld)", columnIndex, columnCount];
	}

	if (_columnsToValues == nil) {
		_columnsToValues = [[NSPointerArray alloc] initWithOptions:NSPointerFunctionsStrongMemory];
		_columnsToValues.count = columnCount;
	}

	[_columnsToValues insertPointer:values atIndex:columnIndex];
}

- (void)buildWithMatrix:(LNKMatrix *)matrix {
	if (_columnsToValues == nil) {
		[NSException raise:NSGenericException format:@"Values must be registered with -registerValues:forColumnAtIndex: prior to calling -buildWithMatrix:."];
	}

	if (matrix.hasBiasColumn) {
		[NSException raise:NSGenericException format:@"Matrices used with a Naive Bayes classifier should not have a bias column"];
	}

	const LNKSize columnCount = matrix.columnCount;

	if (columnCount != self.featureCount) {
		[NSException raise:NSGenericException format:@"The column count of the matrix must be the same as the feature count passed to the initializer"];
	}

	[self _freeBuffers];

	LNKClasses *const classes = self.classes;
	const LNKSize classCount = classes.count;

	_featureProbabilities = calloc(sizeof(LNKFloat *), classCount * columnCount);

	const LNKSize exampleCount = matrix.exampleCount;
	const LNKFloat *const outputVector = matrix.outputVector;
	const BOOL performsLaplacianSmoothing = self.performsLaplacianSmoothing;
	const NSUInteger laplacianSmoothingFactor = self.laplacianSmoothingFactor;

	LNKSize classIndex = 0;

	for (LNKClass *class in classes) {
		// P(c) = # of occurences of c / total number of examples
		const LNKSize outputValue = class.unsignedIntegerValue;
		LNKSize hits = 0;

		for (LNKSize example = 0; example < exampleCount; example++) {
			if (outputVector[example] == outputValue)
				hits++;
		}

		NSUInteger adjustedDenominator = 0;

		if (performsLaplacianSmoothing) {
			hits += laplacianSmoothingFactor;
			adjustedDenominator = laplacianSmoothingFactor * classCount;
		}

		[self _setPrior:(LNKFloat)hits / (exampleCount + adjustedDenominator) forClassAtIndex:classIndex];

		// Calculate P(f_(x,n) | c) for all values n of feature/column x
		for (LNKSize column = 0; column < columnCount; column++) {
			NSArray<NSNumber *> *values = [_columnsToValues pointerAtIndex:column];
			const NSUInteger valuesCount = values.count;

			LNKFloat *valuesVector = LNKFloatCalloc(valuesCount);
			_featureProbabilities[classIndex * columnCount + column] = valuesVector;

			NSUInteger valueIndex = 0;

			for (NSNumber *value in values) {
				const NSUInteger valueUnboxed = value.unsignedIntegerValue;

				for (LNKSize example = 0; example < exampleCount; example++) {
					if (outputVector[example] == outputValue) {
						const LNKFloat *exampleRow = [matrix exampleAtIndex:example];

						if (exampleRow[column] == valueUnboxed)
							valuesVector[valueIndex]++;
					}
				}

				adjustedDenominator = 0;

				if (performsLaplacianSmoothing) {
					valuesVector[valueIndex] += laplacianSmoothingFactor;
					adjustedDenominator = valuesCount * laplacianSmoothingFactor;
				}

				valuesVector[valueIndex] /= (LNKFloat)(hits + adjustedDenominator);
				valueIndex++;
			}
		}
		
		classIndex++;
	}
}

- (LNKFloat)probabilityForClassAtIndex:(LNKSize)classIndex featureAtIndex:(LNKSize)featureIndex value:(LNKFloat)value {
	if (classIndex >= self.classes.count) {
		[NSException raise:NSGenericException format:@"The class index is out-of-bounds"];
	}

	const LNKSize featureCount = self.featureCount;

	if (featureIndex >= featureCount) {
		[NSException raise:NSGenericException format:@"The feature index is out-of-bounds"];
	}

	return _featureProbabilities[classIndex * featureCount + featureIndex][(LNKSize)value];
}

@end
