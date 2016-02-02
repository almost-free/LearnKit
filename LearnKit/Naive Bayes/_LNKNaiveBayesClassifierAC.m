//
//  _LNKNaiveBayesClassifierAC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKNaiveBayesClassifierAC.h"

#import "LNKAccelerate.h"
#import "LNKMatrix.h"
#import "LNKNaiveBayesClassifierPrivate.h"

@implementation _LNKNaiveBayesClassifierAC {
	LNKFloat *_priorProbabilities;
	LNKFloat **_featureProbabilities;
}

- (void)train {
	LNKClasses *classes = self.classes;
	LNKMatrix *matrix = self.matrix;
	
	if (classes.count < 2)
		[NSException raise:NSGenericException format:@"There should be at least two classes"];
	
	if (matrix.hasBiasColumn)
		[NSException raise:NSGenericException format:@"Matrices used with a Naive Bayes classifier should not have a bias column"];
	
	NSPointerArray *columnsToValues = [self _columnsToValues];
	const LNKSize classCount = classes.count;
	const LNKSize exampleCount = matrix.exampleCount;
	const LNKSize columnCount = matrix.columnCount;
	const LNKFloat *outputVector = matrix.outputVector;
	const BOOL performsLaplacianSmoothing = self.performsLaplacianSmoothing;
	const NSUInteger laplacianSmoothingFactor = self.laplacianSmoothingFactor;

	if (_priorProbabilities)
		free(_priorProbabilities);
	
	if (_featureProbabilities)
		free(_featureProbabilities);
	
	_priorProbabilities = LNKFloatCalloc(classCount);
	_featureProbabilities = malloc(sizeof(LNKFloat *) * classCount * columnCount);
	
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

		_priorProbabilities[classIndex] = (LNKFloat)hits / (exampleCount + adjustedDenominator);

		// Calculate P(f_(x,n) | c) for all values n of feature/column x
		for (LNKSize column = 0; column < columnCount; column++) {
			NSArray<NSNumber *> *values = [columnsToValues pointerAtIndex:column];
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

- (id)predictValueForFeatureVector:(LNKVector)featureVector {
	return [self predictValueForFeatureVector:featureVector probability:NULL];
}

- (id)predictValueForFeatureVector:(LNKVector)featureVector probability:(LNKFloat *)outProbability {
	if (!featureVector.data || !featureVector.length)
		[NSException raise:NSGenericException format:@"The feature vector must have a non-zero length"];

	LNKClasses *classes = self.classes;
	const LNKSize columnCount = self.matrix.columnCount;
	LNKSize classIndex = 0;

	LNKClass *bestClass = nil;
	LNKFloat bestLikelihood = LNKFloatMin;

	for (LNKClass *class in classes) {
		// Sum of logarithms:
		//   log(P(c)) + log(P(f_1 | c)) + log(P(f_2 | c)) ... + log(P(f_3 | c))
		// Otherwise:
		//   P(c) * P(f_1 | c) * P(f_2 | c) ... P(f_3 | c)
		const LNKFloat priorProbability = _priorProbabilities[classIndex];
		LNKFloat expectation = LNKLog(priorProbability);

		for (LNKSize column = 0; column < columnCount; column++) {
			const LNKSize featureIndex = featureVector.data[column];
			const LNKFloat probability = _featureProbabilities[classIndex * columnCount + column][featureIndex];

			if (probability == 0) {
				expectation = LNKFloatMin;
				break;
			}

			expectation += LNKLog(probability);
		}

		if (expectation > bestLikelihood) {
			bestLikelihood = expectation;
			bestClass = class;
		}

		classIndex++;
	}

	if (outProbability) {
		const LNKFloat probability = LNK_exp(bestLikelihood);
		*outProbability = bestClass ? probability : 0;
	}

	return bestClass;
}

- (void)dealloc {
	if (_priorProbabilities)
		free(_priorProbabilities);
	
	if (_featureProbabilities)
		free(_featureProbabilities);
	
	[super dealloc];
}

@end
