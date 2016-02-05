//
//  _LNKNaiveBayesClassifierAC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKNaiveBayesClassifierAC.h"

#import "LNKAccelerate.h"
#import "LNKClassProbabilityDistribution.h"
#import "LNKMatrix.h"

@implementation _LNKNaiveBayesClassifierAC

- (void)train {
	[self.probabilityDistribution buildWithMatrix:self.matrix];
}

- (id)predictValueForFeatureVector:(LNKVector)featureVector {
	return [self predictValueForFeatureVector:featureVector probability:NULL];
}

- (id)predictValueForFeatureVector:(LNKVector)featureVector probability:(LNKFloat *)outProbability {
	if (featureVector.data == NULL || featureVector.length == 0) {
		[NSException raise:NSGenericException format:@"The feature vector must have a non-zero length"];
	}

	LNKClasses *const classes = self.classes;
	LNKClassProbabilityDistribution *const probabilityDistribution = self.probabilityDistribution;
	const LNKSize columnCount = self.matrix.columnCount;
	LNKSize classIndex = 0;

	LNKClass *bestClass = nil;
	LNKFloat bestLikelihood = LNKFloatMin;

	for (LNKClass *class in classes) {
		// Sum of logarithms:
		//   log(P(c)) + log(P(f_1 | c)) + log(P(f_2 | c)) ... + log(P(f_3 | c))
		const LNKFloat priorProbability = [probabilityDistribution priorForClassAtIndex:classIndex];
		LNKFloat expectation = LNKLog(priorProbability);

		for (LNKSize column = 0; column < columnCount; column++) {
			const LNKFloat featureValue = featureVector.data[column];
			const LNKFloat probability = [probabilityDistribution probabilityForClassAtIndex:classIndex featureAtIndex:column value:featureValue];

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

@end
