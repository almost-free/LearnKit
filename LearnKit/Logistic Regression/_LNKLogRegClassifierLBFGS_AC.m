//
//  _LNKLogRegClassifierLBFGS_AC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKLogRegClassifierLBFGS_AC.h"

#import "LNKAccelerate.h"
#import "LNKAccelerateGradient.h"
#import "LNKLogRegClassifierPrivate.h"
#import "LNKPredictorPrivate.h"

@implementation _LNKLogRegClassifierLBFGS_AC

- (void)train {
	NSAssert([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmLBFGS class]], @"Unsupported algorithm class");
	LNKOptimizationAlgorithmLBFGS *algorithm = self.algorithm;
	const LNKSize columnCount = self.matrix.columnCount;
	LNKFloat *thetaVector = [self _thetaVector];
	
	LNK_learntheta_lbfgs(self.matrix, thetaVector, algorithm.regularizationEnabled, algorithm.lambda, ^(LNKFloat *h, LNKSize m) {
		LNK_vsigmoid(h, m);
	}, ^(const LNKFloat *theta) {
		LNKFloatCopy(thetaVector, theta, columnCount);
		return [self _evaluateCostFunction];
	});
}

- (id)predictValueForFeatureVector:(LNKVector)featureVector {
	NSParameterAssert(featureVector.data);
	NSParameterAssert(featureVector.length);

	const LNKSize biasOffset = 1;
	
	NSAssert(featureVector.length + biasOffset == self.matrix.columnCount, @"The length of the feature vector must be equal to the number of columns in the matrix");
	// Otherwise, we can't compute the dot product.

	LNKFloat *featuresWithBias = LNKFloatAlloc(featureVector.length + biasOffset);
	featuresWithBias[0] = 1;

	LNKFloatCopy(featuresWithBias + biasOffset, featureVector.data, featureVector.length);

	// sigmoid(theta . input)
	LNKFloat result = 0;
	LNK_dotpr([self _thetaVector], UNIT_STRIDE, featuresWithBias, UNIT_STRIDE, &result, self.matrix.columnCount);
	LNK_vsigmoid(&result, 1);

	free(featuresWithBias);
	
	return [NSNumber numberWithLNKFloat:result];
}

- (LNKFloat)_evaluateCostFunction {
	LNKFloat *thetaVector = [self _thetaVector];
	LNKMatrix *matrix = self.matrix;
	const LNKSize rowCount = matrix.rowCount;
	const LNKSize columnCount = matrix.columnCount;
	const LNKFloat *matrixBuffer = matrix.matrixBuffer;
	const LNKFloat *outputVector = matrix.outputVector;
	
	// 1 / m * sum(-y log(h) - (1 - y) log(1 - h))
	LNKFloat *workgroup = LNKFloatAlloc(rowCount);
	LNK_mmul(matrixBuffer, UNIT_STRIDE, thetaVector, UNIT_STRIDE, workgroup, UNIT_STRIDE, rowCount, 1, columnCount);
	
	// At this point, `workgroup` contains 'h'.
	LNK_vsigmoid(workgroup, rowCount);
	
	const int n = (int)rowCount;
	const LNKFloat one = 1;
	
	LNKFloat *logVector = LNKFloatAlloc(rowCount);
	LNK_vlog(logVector, workgroup, &n);
	
	LNKFloat *negativeOutputVector = LNKFloatAlloc(rowCount);
	LNK_vneg(outputVector, UNIT_STRIDE, negativeOutputVector, UNIT_STRIDE, rowCount);
	
	LNKFloat sum1;
	LNK_dotpr(negativeOutputVector, UNIT_STRIDE, logVector, UNIT_STRIDE, &sum1, rowCount);
	
	// Re-purpose workgroup since it's not used anymore.
	LNKFloat *minusLogVector = workgroup;
	LNK_vneg(minusLogVector, UNIT_STRIDE, minusLogVector, UNIT_STRIDE, rowCount);
	LNK_vsadd(minusLogVector, UNIT_STRIDE, &one, minusLogVector, UNIT_STRIDE, rowCount);
	LNK_vlog(minusLogVector, minusLogVector, &n);
	
	LNK_vsadd(negativeOutputVector, UNIT_STRIDE, &one, negativeOutputVector, UNIT_STRIDE, rowCount);
	
	LNKFloat sum2;
	LNK_dotpr(negativeOutputVector, UNIT_STRIDE, minusLogVector, UNIT_STRIDE, &sum2, rowCount);
	
	free(logVector);
	free(minusLogVector);
	free(negativeOutputVector);
	
	LNKFloat cost = (sum1 - sum2) / rowCount;
	
	NSAssert([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmLBFGS class]], @"Unsupported algorithm class");
	LNKOptimizationAlgorithmLBFGS *algorithm = self.algorithm;
	
	if (algorithm.regularizationEnabled) {
		// 1/2 lambda / m * sum(pow(theta, 2))
		const LNKFloat regularizationFactor = algorithm.lambda * 0.5 / rowCount;
		
		// Don't regularize the first parameter.
		const LNKFloat previousFirstValue = thetaVector[0];
		thetaVector[0] = 0;
		
		LNKFloat thetaSum;
		LNK_dotpr(thetaVector, UNIT_STRIDE, thetaVector, UNIT_STRIDE, &thetaSum, columnCount);
		
		thetaVector[0] = previousFirstValue;
		
		cost += regularizationFactor * thetaSum;
	}
	
	return cost;
}

- (LNKFloat)computeClassificationAccuracyOnMatrix:(LNKMatrix *)matrix {
	LNKFloat *thetaVector = [self _thetaVector];
	
	const LNKSize rowCount = matrix.rowCount;
	const LNKSize columnCount = matrix.columnCount;
	const LNKFloat *matrixBuffer = matrix.matrixBuffer;
	const LNKFloat *outputVector = matrix.outputVector;
	
	LNKFloat *workgroup = LNKFloatAlloc(rowCount);
	LNK_mmul(matrixBuffer, UNIT_STRIDE, thetaVector, UNIT_STRIDE, workgroup, UNIT_STRIDE, rowCount, 1, columnCount);
	
	LNKSize hits = 0;
	
	// With a sigmoid function, y=1 when X . theta > 0
	for (LNKSize m = 0; m < rowCount; m++) {
		if ((workgroup[m] > 0 && outputVector[m] == 1) || (workgroup[m] <= 0 && outputVector[m] == 0))
			hits++;
	}
	
	free(workgroup);
	
	return (LNKFloat)hits / rowCount;
}

@end
