//
//  _LNKLinRegPredictorAC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKLinRegPredictorAC.h"

#import "LNKAccelerate.h"
#import "LNKLinRegPredictorPrivate.h"
#import "LNKMatrix.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

@implementation _LNKLinRegPredictorAC

- (id)predictValueForFeatureVector:(const LNKFloat *)featureVector length:(LNKSize)length {
	NSParameterAssert(featureVector);
	NSParameterAssert(length);
	
	NSAssert(length == self.matrix.columnCount, @"The length of the feature vector must be equal to the number of columns in the matrix");
	// Otherwise, we can't compute the dot product.
	
	LNKMatrix *matrix = self.matrix;
	LNKFloat *thetaVector = [self _thetaVector];
	const LNKSize columnCount = matrix.columnCount;
	LNKFloat *featureVectorNormalizedIfNeeded = (LNKFloat *)featureVector;
	
	if (matrix.normalized) {
		// We need to copy the input vector since -normalizeVector: works in-place.
		featureVectorNormalizedIfNeeded = LNKFloatAlloc(columnCount);
		LNKFloatCopy(featureVectorNormalizedIfNeeded, featureVector, columnCount);
		
		[matrix normalizeVector:featureVectorNormalizedIfNeeded];
	}
	
	LNKFloat result;
	LNK_dotpr(thetaVector, UNIT_STRIDE, featureVectorNormalizedIfNeeded, UNIT_STRIDE, &result, columnCount);
	
	if (featureVectorNormalizedIfNeeded != featureVector)
		free(featureVectorNormalizedIfNeeded);
	
	return [NSNumber numberWithLNKFloat:result];
}

- (LNKFloat)_evaluateCostFunction {
	LNKFloat *thetaVector = [self _thetaVector];
	LNKMatrix *matrix = self.matrix;
	const LNKSize exampleCount = matrix.exampleCount;
	const LNKSize columnCount = matrix.columnCount;
	
	// 1 / (2 m) * sum(pow(h - y, 2))
	const LNKFloat factor = 0.5 / exampleCount;
	LNKFloat *workgroup = LNKFloatAlloc(exampleCount);
	
	LNK_mmul(matrix.matrixBuffer, UNIT_STRIDE, thetaVector, UNIT_STRIDE, workgroup, UNIT_STRIDE, exampleCount, 1, columnCount);
	LNK_vsub(workgroup, UNIT_STRIDE, matrix.outputVector, UNIT_STRIDE, workgroup, UNIT_STRIDE, exampleCount);
	LNKFloat sum;
	LNK_dotpr(workgroup, UNIT_STRIDE, workgroup, UNIT_STRIDE, &sum, exampleCount);
	free(workgroup);
	
	sum *= factor;
	
	if ([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmRegularizable class]]) {
		LNKOptimizationAlgorithmRegularizable *algorithm = self.algorithm;
		
		if (algorithm.regularizationEnabled) {
			// Regularization: += 0.5 * lambda / m * sum(Theta^2)
			const LNKSize skipBiasUnit = 1;
			LNKFloat thetaSum;
			LNK_dotpr(thetaVector + skipBiasUnit, UNIT_STRIDE, thetaVector + skipBiasUnit, UNIT_STRIDE, &thetaSum, columnCount - skipBiasUnit);
			sum += 0.5 * algorithm.lambda / exampleCount * thetaSum;
		}
	}
	
	return sum;
}

@end
