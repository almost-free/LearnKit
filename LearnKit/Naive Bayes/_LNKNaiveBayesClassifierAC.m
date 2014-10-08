//
//  _LNKNaiveBayesClassifierAC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKNaiveBayesClassifierAC.h"

#import "LNKAccelerate.h"
#import "LNKDesignMatrix.h"
#import "LNKNaiveBayesClassifierPrivate.h"

@implementation _LNKNaiveBayesClassifierAC {
	LNKFloat *_priorProbabilities;
	LNKFloat **_featureProbabilities;
}

- (void)train {
	LNKClasses *classes = self.classes;
	LNKDesignMatrix *designMatrix = self.designMatrix;
	
	if (classes.count < 2) {
		@throw [NSException exceptionWithName:NSGenericException reason:@"There should be at least two classes" userInfo:nil];
	}
	
	if (designMatrix.hasBiasColumn) {
		@throw [NSException exceptionWithName:NSGenericException reason:@"Design matrices used with a Naive Bayes classifier should not have a bias column" userInfo:nil];
	}
	
	NSPointerArray *columnsToValues = [self _columnsToValues];
	const LNKSize classCount = classes.count;
	const LNKSize exampleCount = designMatrix.exampleCount;
	const LNKSize columnCount = designMatrix.columnCount;
	const LNKFloat *outputVector = designMatrix.outputVector;
	
	if (_priorProbabilities)
		free(_priorProbabilities);
	
	if (_featureProbabilities)
		free(_featureProbabilities);
	
	_priorProbabilities = LNKFloatCalloc(classCount);
	_featureProbabilities = malloc(sizeof(LNKFloat *) * classCount * columnCount);
	
	LNKSize classIndex = 0;
	
	for (LNKClass *class in classes) {
		const LNKSize outputValue = class.unsignedIntegerValue;
		LNKSize hits = 0;
		
		for (LNKSize example = 0; example < exampleCount; example++) {
			if (outputVector[example] == outputValue)
				hits++;
		}
		
		_priorProbabilities[classIndex] = (LNKFloat)hits / exampleCount;
		
		for (LNKSize column = 0; column < columnCount; column++) {
			NSArray *values = [columnsToValues pointerAtIndex:column];
			
			LNKFloat *valuesVector = LNKFloatCalloc(values.count);
			_featureProbabilities[classIndex * columnCount + column] = valuesVector;
			
			NSUInteger valueIndex = 0;
			
			for (NSNumber *value in values) {
				for (LNKSize example = 0; example < exampleCount; example++) {
					if (outputVector[example] == outputValue) {
						const LNKFloat *exampleRow = [designMatrix exampleAtIndex:example];
						
						if (exampleRow[column] == value.unsignedIntegerValue) {
							valuesVector[valueIndex]++;
						}
					}
				}
				
				valuesVector[valueIndex] /= (LNKFloat)hits;
				valueIndex++;
			}
		}
		
		classIndex++;
	}
}

- (id)predictValueForFeatureVector:(const LNKFloat *)featureVector length:(LNKSize)length {
	NSParameterAssert(featureVector);
	NSParameterAssert(length);
	
	LNKClasses *classes = self.classes;
	const LNKSize columnCount = self.designMatrix.columnCount;
	LNKSize classIndex = 0;
	
	LNKClass *bestClass = nil;
	LNKFloat bestLikelihood = -1;
	
	for (LNKClass *class in classes) {
		LNKFloat expectation = _priorProbabilities[classIndex];
		
		for (LNKSize column = 0; column < columnCount; column++) {
			const LNKSize featureIndex = featureVector[column];
			
			expectation *= _featureProbabilities[classIndex * columnCount + column][featureIndex];
		}
		
		if (expectation > bestLikelihood) {
			bestLikelihood = expectation;
			bestClass = class;
		}
		
		classIndex++;
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
