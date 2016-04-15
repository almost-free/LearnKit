//
//  LNKGaussianProbabilityDistribution.m
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKGaussianProbabilityDistribution.h"

#import "LNKAccelerate.h"
#import "LNKClasses.h"
#import "LNKClassProbabilityDistributionPrivate.h"
#import "LNKMatrix.h"

typedef struct {
	LNKFloat mean;
	LNKFloat sd;
} LNKGaussianParameters;

@implementation LNKGaussianProbabilityDistribution {
	LNKGaussianParameters *_featureParameters;
}

- (void)dealloc {
	[self _freeBuffers];
	[super dealloc];
}

- (void)_freeBuffers {
	if (_featureParameters != NULL) {
		free(_featureParameters);
		_featureParameters = NULL;
	}
}

- (void)buildWithMatrix:(LNKMatrix *)matrix {
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
	_featureParameters = calloc(columnCount * classCount, sizeof(LNKGaussianParameters));

	const LNKSize rowCount = matrix.rowCount;
	const LNKFloat *const outputVector = matrix.outputVector;

	LNKSize classIndex = 0;

	for (LNKClass *class in classes) {
		// P(c) = # of occurences of c / total number of examples
		const LNKSize outputValue = class.unsignedIntegerValue;
		LNKSize hits = 0;

		for (LNKSize row = 0; row < rowCount; row++) {
			if (LNK_fabs(outputVector[row] - (LNKFloat)outputValue) < FLT_MIN) {
				hits++;
			}
		}

		[self _setPrior:(LNKFloat)hits / rowCount forClassAtIndex:classIndex];

		// Calculate mean and SD for all features and classes.
		for (LNKSize column = 0; column < columnCount; column++) {
			LNKGaussianParameters *const parameters = &_featureParameters[classIndex * columnCount + column];

			const LNKVector vector = [matrix copyOfColumnAtIndex:column];
			LNKFloat mean = 0;
			LNK_vmean(vector.data, UNIT_STRIDE, &mean, vector.length);

			const LNKFloat sd = LNK_vsd(vector, UNIT_STRIDE, NULL, mean, NO);

			parameters->mean = mean;
			parameters->sd = sd;

			LNKVectorRelease(vector);
		}
	}
}

- (LNKFloat)probabilityLogForClassAtIndex:(LNKSize)classIndex featureAtIndex:(LNKSize)featureIndex value:(LNKFloat)value {
	if (classIndex >= self.classes.count) {
		[NSException raise:NSGenericException format:@"The class index is out-of-bounds"];
	}

	const LNKSize featureCount = self.featureCount;

	if (featureIndex >= featureCount) {
		[NSException raise:NSGenericException format:@"The feature index is out-of-bounds"];
	}

	const LNKGaussianParameters parameters = _featureParameters[classIndex * featureCount + featureIndex];
	const LNKFloat top = value - parameters.mean;
	const LNKFloat bottom = parameters.sd * parameters.sd;
	return -0.5 * top * top / bottom - LNKLog(parameters.sd);
}

@end
