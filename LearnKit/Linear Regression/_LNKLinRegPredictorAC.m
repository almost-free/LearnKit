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

- (id)predictValueForFeatureVector:(LNKVector)featureVector {
	NSParameterAssert(featureVector.data);
	NSParameterAssert(featureVector.length);

	const LNKSize biasOffset = 1;

	NSAssert(featureVector.length + biasOffset == self.matrix.columnCount, @"The length of the feature vector must be equal to the number of columns in the matrix");
	// Otherwise, we can't compute the dot product.

	LNKFloat *featuresWithBias = LNKFloatAlloc(featureVector.length + biasOffset);
	featuresWithBias[0] = 1;
	LNKFloatCopy(featuresWithBias + biasOffset, featureVector.data, featureVector.length);
	
	LNKMatrix *matrix = self.matrix;
	LNKFloat *thetaVector = [self _thetaVector];
	const LNKSize columnCount = matrix.columnCount;
	
	if (matrix.normalized) {
		[matrix normalizeVector:featuresWithBias];
	}
	
	LNKFloat result = 0;
	LNK_dotpr(thetaVector, UNIT_STRIDE, featuresWithBias, UNIT_STRIDE, &result, columnCount + biasOffset);
	
	free(featuresWithBias);
	
	return [NSNumber numberWithLNKFloat:result];
}

- (LNKFloat)_evaluateCostFunction {
	LNKFloat *thetaVector = [self _thetaVector];
	LNKMatrix *matrix = self.matrix;
	const LNKSize rowCount = matrix.rowCount;
	const LNKSize columnCount = matrix.columnCount;
	
	// 1 / (2 m) * sum(pow(h - y, 2))
	const LNKFloat factor = 0.5 / rowCount;
	LNKFloat *workgroup = LNKFloatAlloc(rowCount);
	
	LNK_mmul(matrix.matrixBuffer, UNIT_STRIDE, thetaVector, UNIT_STRIDE, workgroup, UNIT_STRIDE, rowCount, 1, columnCount);
	LNK_vsub(workgroup, UNIT_STRIDE, matrix.outputVector, UNIT_STRIDE, workgroup, UNIT_STRIDE, rowCount);
	LNKFloat sum;
	LNK_dotpr(workgroup, UNIT_STRIDE, workgroup, UNIT_STRIDE, &sum, rowCount);
	free(workgroup);
	
	sum *= factor;
	
	if ([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmRegularizable class]]) {
		LNKOptimizationAlgorithmRegularizable *algorithm = self.algorithm;
		
		if (algorithm.regularizationEnabled) {
			// Regularization: += 0.5 * lambda / m * sum(Theta^2)
			const LNKSize skipBiasUnit = 1;
			LNKFloat thetaSum;
			LNK_dotpr(thetaVector + skipBiasUnit, UNIT_STRIDE, thetaVector + skipBiasUnit, UNIT_STRIDE, &thetaSum, columnCount - skipBiasUnit);
			sum += 0.5 * algorithm.lambda / rowCount * thetaSum;
		}
	}
	
	return sum;
}

@end
