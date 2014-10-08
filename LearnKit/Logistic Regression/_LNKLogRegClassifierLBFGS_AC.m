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
	const LNKSize columnCount = self.designMatrix.columnCount;
	LNKFloat *thetaVector = [self _thetaVector];
	
	LNK_learntheta_lbfgs(self.designMatrix, thetaVector, algorithm.regularizationEnabled, algorithm.lambda, ^(LNKFloat *h, LNKSize m) {
		LNK_vsigmoid(h, m);
	}, ^(const LNKFloat *theta) {
		LNKFloatCopy(thetaVector, theta, columnCount);
		return [self _evaluateCostFunction];
	});
}

- (id)predictValueForFeatureVector:(const LNKFloat *)featureVector length:(LNKSize)length {
	NSParameterAssert(featureVector);
	NSParameterAssert(length);
	
	NSAssert(length == self.designMatrix.columnCount, @"The length of the feature vector must be equal to the number of columns in the design matrix");
	// Otherwise, we can't compute the dot product.
	
	// sigmoid(theta . input)
	LNKFloat result;
	LNK_dotpr([self _thetaVector], UNIT_STRIDE, featureVector, UNIT_STRIDE, &result, self.designMatrix.columnCount);
	LNK_vsigmoid(&result, 1);
	
	return [NSNumber numberWithLNKFloat:result];
}

- (LNKFloat)_evaluateCostFunction {
	LNKFloat *thetaVector = [self _thetaVector];
	LNKDesignMatrix *designMatrix = self.designMatrix;
	const LNKSize exampleCount = designMatrix.exampleCount;
	const LNKSize columnCount = designMatrix.columnCount;
	const LNKFloat *matrix = designMatrix.matrixBuffer;
	const LNKFloat *outputVector = designMatrix.outputVector;
	
	// 1 / m * sum(-y log(h) - (1 - y) log(1 - h))
	LNKFloat *workgroup = LNKFloatAlloc(exampleCount);
	LNK_mmul(matrix, UNIT_STRIDE, thetaVector, UNIT_STRIDE, workgroup, UNIT_STRIDE, exampleCount, 1, columnCount);
	
	// At this point, `workgroup` contains 'h'.
	LNK_vsigmoid(workgroup, exampleCount);
	
	const int n = (int)exampleCount;
	const LNKFloat one = 1;
	
	LNKFloat *logVector = LNKFloatAlloc(exampleCount);
	LNK_vlog(logVector, workgroup, &n);
	
	LNKFloat *negativeOutputVector = LNKFloatAlloc(exampleCount);
	LNK_vneg(outputVector, UNIT_STRIDE, negativeOutputVector, UNIT_STRIDE, exampleCount);
	
	LNKFloat sum1;
	LNK_dotpr(negativeOutputVector, UNIT_STRIDE, logVector, UNIT_STRIDE, &sum1, exampleCount);
	
	// Re-purpose workgroup since it's not used anymore.
	LNKFloat *minusLogVector = workgroup;
	LNK_vneg(minusLogVector, UNIT_STRIDE, minusLogVector, UNIT_STRIDE, exampleCount);
	LNK_vsadd(minusLogVector, UNIT_STRIDE, &one, minusLogVector, UNIT_STRIDE, exampleCount);
	LNK_vlog(minusLogVector, minusLogVector, &n);
	
	LNK_vsadd(negativeOutputVector, UNIT_STRIDE, &one, negativeOutputVector, UNIT_STRIDE, exampleCount);
	
	LNKFloat sum2;
	LNK_dotpr(negativeOutputVector, UNIT_STRIDE, minusLogVector, UNIT_STRIDE, &sum2, exampleCount);
	
	free(logVector);
	free(minusLogVector);
	free(negativeOutputVector);
	
	LNKFloat cost = (sum1 - sum2) / exampleCount;
	
	NSAssert([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmLBFGS class]], @"Unsupported algorithm class");
	LNKOptimizationAlgorithmLBFGS *algorithm = self.algorithm;
	
	if (algorithm.regularizationEnabled) {
		// 1/2 lambda / m * sum(pow(theta, 2))
		const LNKFloat regularizationFactor = algorithm.lambda * 0.5 / exampleCount;
		
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

- (LNKFloat)computeClassificationAccuracy {
	LNKFloat *thetaVector = [self _thetaVector];
	LNKDesignMatrix *designMatrix = self.designMatrix;
	const LNKSize exampleCount = designMatrix.exampleCount;
	const LNKSize columnCount = designMatrix.columnCount;
	const LNKFloat *matrix = designMatrix.matrixBuffer;
	const LNKFloat *outputVector = designMatrix.outputVector;
	
	LNKFloat *workgroup = LNKFloatAlloc(exampleCount);
	LNK_mmul(matrix, UNIT_STRIDE, thetaVector, UNIT_STRIDE, workgroup, UNIT_STRIDE, exampleCount, 1, columnCount);
	
	LNKSize hits = 0;
	
	// With a sigmoid function, y=1 when X . theta > 0
	for (LNKSize m = 0; m < exampleCount; m++) {
		if ((workgroup[m] > 0 && outputVector[m] == 1) || (workgroup[m] <= 0 && outputVector[m] == 0))
			hits++;
	}
	
	free(workgroup);
	
	return (LNKFloat)hits / exampleCount;
}

@end
