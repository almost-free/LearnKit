//
//  _LNKAnomalyDetectorAC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKAnomalyDetectorAC.h"

#import "LNKAccelerate.h"
#import "LNKMatrix.h"

@implementation _LNKAnomalyDetectorAC {
	LNKFloat *_mu;
	LNKFloat *_sigma2;
}

- (void)train {
	LNKMatrix *matrix = self.matrix;
	const LNKSize columnCount = matrix.columnCount;
	const LNKSize rowCount = matrix.rowCount;
	const LNKFloat *matrixBuffer = matrix.matrixBuffer;
	
	_mu = LNKFloatAlloc(columnCount);
	_sigma2 = LNKFloatCalloc(columnCount * columnCount);
	
	LNKFloat *workgroup = LNKFloatAlloc(rowCount);
	
	// Compute Gaussian parameters.
	for (LNKSize column = 0; column < columnCount; column++) {
		const LNKFloat *columnPointer = matrixBuffer + column;
		
		LNKFloat mean;
		LNK_vmean(columnPointer, columnCount, &mean, rowCount);
		
		_mu[column] = -mean;
		
		// `sigma2` is actually a (diagonal) covariance matrix.
		const LNKVector inVector = LNKVectorCreateUnsafe(columnPointer, rowCount);
		_sigma2[column * columnCount + column] = LNK_pow(LNK_vsd(inVector, columnCount, workgroup, mean, NO), 2);
	}
	
	free(workgroup);
}

- (LNKFloat *)_muVector {
	return _mu;
}

- (LNKFloat *)_sigmaMatrix {
	return _sigma2;
}

- (void)_setMuVector:(LNKFloat *)vector {
	NSParameterAssert(vector);
	
	const LNKSize columnCount = self.matrix.columnCount;
	_mu = LNKFloatAllocAndCopy(vector, columnCount);
}

- (void)_setSigmaMatrix:(LNKFloat *)matrix {
	NSParameterAssert(matrix);
	
	const LNKSize columnCount = self.matrix.columnCount;
	_sigma2 = LNKFloatAllocAndCopy(matrix, columnCount * columnCount);
}

- (LNKFloat)_probabilityWithFeatureVector:(const LNKFloat *)featureVector length:(LNKSize)length {
#pragma unused(length)
	
	NSParameterAssert(featureVector);
	NSParameterAssert(length);
	
	LNKMatrix *matrix = self.matrix;
	const LNKSize columnCount = matrix.columnCount;
	
	// (2pi)^(-n/2) * det(sigma2)^(-0.5) ...
	const LNKFloat det = LNK_mdet(_sigma2, columnCount);
	
	if (det == 0)
		return 0;
	
	const LNKFloat c = LNK_pow(2 * M_PI, -(LNKFloat)columnCount/2) * LNK_pow(det, -0.5);
	
	// ... * e^(-0.5 * sum((featureVector * pinv(sigma2)) * featureVector))
	LNKFloat *sigmaInverse = LNKFloatAllocAndCopy(_sigma2, columnCount * columnCount);
	LNK_minvert(sigmaInverse, columnCount);
	
	// Normalize the feature vector about `mu`.
	LNKFloat *featureVectorAdjusted = LNKFloatAlloc(columnCount);
	LNK_vadd(featureVector, UNIT_STRIDE, _mu, UNIT_STRIDE, featureVectorAdjusted, UNIT_STRIDE, columnCount);
	
	LNKFloat *results = LNKFloatAlloc(columnCount);
	LNK_mmul(featureVectorAdjusted, UNIT_STRIDE, sigmaInverse, UNIT_STRIDE, results, UNIT_STRIDE, 1, columnCount, columnCount);
	free(sigmaInverse);
	
	LNK_vmul(results, UNIT_STRIDE, featureVectorAdjusted, UNIT_STRIDE, results, UNIT_STRIDE, columnCount);
	free(featureVectorAdjusted);
	
	LNKFloat sum;
	LNK_vsum(results, UNIT_STRIDE, &sum, columnCount);
	free(results);
	
	return c * LNK_exp(-0.5 * sum);
}

- (id)predictValueForFeatureVector:(LNKVector)featureVector {
	const LNKFloat p = [self _probabilityWithFeatureVector:featureVector.data length:featureVector.length];
	
	if (p < self.threshold) {
		// It's an anomaly.
		return [LNKClass classWithUnsignedInteger:1];
	}
	
	return [LNKClass classWithUnsignedInteger:0];
}

- (void)dealloc {
	if (_mu)
		free(_mu);
	
	if (_sigma2)
		free(_sigma2);
	
	[super dealloc];
}

@end
