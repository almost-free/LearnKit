//
//  LNKCollaborativeFilteringPredictor.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKCollaborativeFilteringPredictor.h"

#import "LNKAccelerate.h"
#import "LNKCollaborativeFilteringPredictorPrivate.h"
#import "LNKMatrix.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

@implementation LNKCollaborativeFilteringPredictor {
	LNKSize _userCount;
	LNKFloat *_thetaMatrix;
}

+ (NSArray *)supportedAlgorithms {
	return @[ [LNKOptimizationAlgorithmCG class] ];
}

+ (NSArray *)supportedImplementationTypes {
	return @[ @(LNKImplementationTypeAccelerate) ];
}

- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm userCount:(NSUInteger)userCount {
#pragma unused(implementationType)
	
	if (userCount == 0)
		[NSException raise:NSGenericException format:@"The user count must be greater than 0"];
	
	self = [self initWithMatrix:matrix optimizationAlgorithm:algorithm];
	if (self) {
		_userCount = userCount;
	}
	return self;
}

#warning TODO: need pre-flight check if indicatorMatrix and outputMatrix is nil

- (LNKFloat)_evaluateCostFunction {
	LNKMatrix *matrix = self.matrix;
	const LNKSize featureCount = matrix.columnCount;
	const LNKSize exampleCount = matrix.exampleCount;
	const LNKFloat *dataMatrix = matrix.matrixBuffer;
	
	LNKFloat *thetaTranspose = LNKFloatAlloc(_userCount * featureCount);
	LNK_mtrans(_thetaMatrix, UNIT_STRIDE, thetaTranspose, UNIT_STRIDE, featureCount, _userCount);
	
	// 1/2 * sum((((X * Theta') - Y) ^ 2) * R)
	const LNKSize resultSize = exampleCount * _userCount;
	LNKFloat *result = LNKFloatAlloc(resultSize);
	LNK_mmul(dataMatrix, UNIT_STRIDE, thetaTranspose, UNIT_STRIDE, result, UNIT_STRIDE, exampleCount, _userCount, featureCount);
	
	LNK_vsub(_outputMatrix.matrixBuffer, UNIT_STRIDE, result, UNIT_STRIDE, result, UNIT_STRIDE, resultSize);
	LNK_vmul(result, UNIT_STRIDE, result, UNIT_STRIDE, result, UNIT_STRIDE, resultSize);
	LNK_vmul(result, UNIT_STRIDE, _indicatorMatrix.matrixBuffer, UNIT_STRIDE, result, UNIT_STRIDE, resultSize);
	
	LNKFloat sum;
	LNK_vsum(result, UNIT_STRIDE, &sum, resultSize);
	free(result);
	
	LNKFloat regularizationTerm = 0;
	
	NSAssert([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmCG class]], @"Unexpected algorithm");
	LNKOptimizationAlgorithmCG *algorithm = self.algorithm;
	
	if (algorithm.regularizationEnabled) {
		// Re-use the theta transpose matrix to compute the theta square.
		LNK_vmul(thetaTranspose, UNIT_STRIDE, thetaTranspose, UNIT_STRIDE, thetaTranspose, UNIT_STRIDE, _userCount * featureCount);
		
		LNKFloat *dataSquare = LNKFloatAlloc(exampleCount * featureCount);
		LNK_vmul(dataMatrix, UNIT_STRIDE, dataMatrix, UNIT_STRIDE, dataSquare, UNIT_STRIDE, exampleCount * featureCount);
		
		LNKFloat thetaSum, dataSum;
		LNK_vsum(thetaTranspose, UNIT_STRIDE, &thetaSum, _userCount * featureCount);
		LNK_vsum(dataSquare, UNIT_STRIDE, &dataSum, exampleCount * featureCount);
		free(dataSquare);
		
		// ... + lambda / 2 * (sum(Theta^2) + sum(X^2))
		regularizationTerm = algorithm.lambda / 2 * (thetaSum + dataSum);
	}
	
	free(thetaTranspose);
	
	return 0.5 * sum + regularizationTerm;
}

- (const LNKFloat *)_computeGradient {
	LNKMatrix *matrix = self.matrix;
	const LNKSize featureCount = matrix.columnCount;
	const LNKSize exampleCount = matrix.exampleCount;
	const LNKSize unrolledExampleCount = exampleCount + _userCount;
	
	LNKFloat *unrolledGradient = LNKFloatCalloc(unrolledExampleCount * featureCount);
	
	LNKFloat *dataGradient = unrolledGradient;
	LNKFloat *thetaGradient = unrolledGradient + exampleCount * featureCount;
	
	NSAssert([self.algorithm isKindOfClass:[LNKOptimizationAlgorithmCG class]], @"Unexpected algorithm");
	LNKOptimizationAlgorithmCG *algorithm = self.algorithm;
	const BOOL regularizationEnabled = algorithm.regularizationEnabled;
	const LNKFloat lambda = algorithm.lambda;
	
	LNKFloat *workspace = LNKFloatAlloc(featureCount);
	
	for (LNKSize exampleIndex = 0; exampleIndex < exampleCount; exampleIndex++) {
		const LNKFloat *example = [matrix exampleAtIndex:exampleIndex];
		const LNKFloat *output = [_outputMatrix exampleAtIndex:exampleIndex];
		const LNKFloat *indicator = [_indicatorMatrix exampleAtIndex:exampleIndex];
		
		LNKFloat *dataGradientLocation = dataGradient + exampleIndex * featureCount;
		
		for (LNKSize userIndex = 0; userIndex < _userCount; userIndex++) {
			if (indicator[userIndex]) {
				// inner = (X(example,:) . Theta(user,:)) - Y(example,user)
				const LNKFloat *user = _thetaMatrix + userIndex * featureCount;
				
				LNKFloat result;
				LNK_dotpr(example, UNIT_STRIDE, user, UNIT_STRIDE, &result, featureCount);
				
				const LNKFloat inner = result - output[userIndex];
				
				// X_gradient += inner * Theta(user,:)
				LNKFloatCopy(workspace, user, featureCount);
				LNK_vsmul(workspace, UNIT_STRIDE, &inner, workspace, UNIT_STRIDE, featureCount);
				
				LNK_vadd(dataGradientLocation, UNIT_STRIDE, workspace, UNIT_STRIDE, dataGradientLocation, UNIT_STRIDE, featureCount);
				
				// Theta_gradient += inner * X(example,:)
				LNKFloatCopy(workspace, example, featureCount);
				LNK_vsmul(workspace, UNIT_STRIDE, &inner, workspace, UNIT_STRIDE, featureCount);
				
				LNKFloat *thetaGradientLocation = thetaGradient + userIndex * featureCount;
				LNK_vadd(thetaGradientLocation, UNIT_STRIDE, workspace, UNIT_STRIDE, thetaGradientLocation, UNIT_STRIDE, featureCount);
			}
		}
		
		if (regularizationEnabled) {
			// X_gradient(example,:) += lambda * X(example,:)
			LNKFloatCopy(workspace, example, featureCount);
			LNK_vsmul(workspace, UNIT_STRIDE, &lambda, workspace, UNIT_STRIDE, featureCount);
			LNK_vadd(dataGradientLocation, UNIT_STRIDE, workspace, UNIT_STRIDE, dataGradientLocation, UNIT_STRIDE, featureCount);
		}
	}
	
	if (regularizationEnabled) {
		for (LNKSize userIndex = 0; userIndex < _userCount; userIndex++) {
			LNKFloat *thetaGradientLocation = thetaGradient + userIndex * featureCount;
			
			// Theta_gradient(user,:) += lambda * Theta(user,:)
			LNKFloatCopy(workspace, _thetaMatrix + userIndex * featureCount, featureCount);
			LNK_vsmul(workspace, UNIT_STRIDE, &lambda, workspace, UNIT_STRIDE, featureCount);
			LNK_vadd(thetaGradientLocation, UNIT_STRIDE, workspace, UNIT_STRIDE, thetaGradientLocation, UNIT_STRIDE, featureCount);
		}
	}
	
	free(workspace);
	
	return unrolledGradient;
}

- (void)_setThetaMatrix:(LNKMatrix *)matrix {
	NSParameterAssert(matrix);
	
	if (_thetaMatrix)
		free(_thetaMatrix);
	
	_thetaMatrix = LNKFloatAllocAndCopy(matrix.matrixBuffer, _userCount * matrix.columnCount);
}

- (void)dealloc {
	free(_thetaMatrix);
	
	[_outputMatrix release];
	[_indicatorMatrix release];
	
	[super dealloc];
}

@end
