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
	LNKFloat *_thetaVector;
	LNKFloat *_indicatorMatrix;
	LNKFloat *_outputMatrix;
}

+ (NSArray *)supportedAlgorithms {
	return @[ [LNKOptimizationAlgorithmCG class] ];
}

+ (NSArray *)supportedImplementationTypes {
	return @[ @(LNKImplementationTypeAccelerate) ];
}

- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementationType optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm userCount:(NSUInteger)userCount {
#pragma unused(implementationType)
	
	if (userCount == 0) {
		[NSException raise:NSGenericException format:@"The user count must be greater than 0"];
	}
	
	self = [self initWithMatrix:matrix optimizationAlgorithm:algorithm];
	if (self) {
		_userCount = userCount;
	}
	return self;
}

- (LNKFloat)_evaluateCostFunction {
	LNKMatrix *matrix = self.matrix;
	const LNKSize featureCount = matrix.columnCount;
	const LNKSize exampleCount = matrix.exampleCount;
	const LNKFloat *dataMatrix = matrix.matrixBuffer;
	
	LNKFloat *thetaTranspose = LNKFloatAlloc(_userCount * featureCount);
	LNK_mtrans(_thetaVector, UNIT_STRIDE, thetaTranspose, UNIT_STRIDE, featureCount, _userCount);
	
	const LNKSize resultSize = exampleCount * _userCount;
	LNKFloat *result = LNKFloatAlloc(resultSize);
	LNK_mmul(dataMatrix, UNIT_STRIDE, thetaTranspose, UNIT_STRIDE, result, UNIT_STRIDE, exampleCount, _userCount, featureCount);
	
	LNK_vsub(_outputMatrix, UNIT_STRIDE, result, UNIT_STRIDE, result, UNIT_STRIDE, resultSize);
	LNK_vmul(result, UNIT_STRIDE, result, UNIT_STRIDE, result, UNIT_STRIDE, resultSize);
	LNK_vmul(result, UNIT_STRIDE, _indicatorMatrix, UNIT_STRIDE, result, UNIT_STRIDE, resultSize);
	
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
		
		regularizationTerm = algorithm.lambda / 2 * (thetaSum + dataSum);
	}
	
	free(thetaTranspose);
	
	return 0.5 * sum + regularizationTerm;
}

- (void)_copyThetaVector:(const LNKFloat *)vector shouldTranspose:(BOOL)shouldTranspose {
	NSParameterAssert(vector);
	
	const LNKSize columnCount = self.matrix.columnCount;
	const LNKSize size = _userCount * columnCount;
	
	_thetaVector = LNKFloatAlloc(size);
	
	if (shouldTranspose)
		LNK_mtrans(vector, UNIT_STRIDE, _thetaVector, UNIT_STRIDE, _userCount, columnCount);
	else
		LNKFloatCopy(_thetaVector, vector, size);
}

- (void)copyIndicatorMatrix:(const LNKFloat *)matrix shouldTranspose:(BOOL)shouldTranspose {
	NSParameterAssert(matrix);
	
	const LNKSize exampleCount = self.matrix.exampleCount;
	const LNKSize size = exampleCount * _userCount;
	
	_indicatorMatrix = LNKFloatAlloc(size);
	
	if (shouldTranspose)
		LNK_mtrans(matrix, UNIT_STRIDE, _indicatorMatrix, UNIT_STRIDE, exampleCount, _userCount);
	else
		LNKFloatCopy(_indicatorMatrix, matrix, size);
}

- (void)copyOutputMatrix:(const LNKFloat *)matrix shouldTranspose:(BOOL)shouldTranspose {
	NSParameterAssert(matrix);
	
	const LNKSize exampleCount = self.matrix.exampleCount;
	const LNKSize size = exampleCount * _userCount;
	
	_outputMatrix = LNKFloatAlloc(size);
	
	if (shouldTranspose)
		LNK_mtrans(matrix, UNIT_STRIDE, _outputMatrix, UNIT_STRIDE, exampleCount, _userCount);
	else
		LNKFloatCopy(_outputMatrix, matrix, size);
}

- (void)dealloc {
	free(_thetaVector);
	free(_outputMatrix);
	free(_indicatorMatrix);
	
	[super dealloc];
}

@end