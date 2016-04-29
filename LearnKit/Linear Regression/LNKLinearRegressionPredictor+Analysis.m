//
//  LNKLinearRegressionPredictor+Analysis.m
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKLinearRegressionPredictor+Analysis.h"

#import "LNKAccelerate.h"
#import "LNKLinearRegressionPredictorPrivate.h"
#import "LNKMatrix.h"

@implementation LNKLinearRegressionPredictor (Analysis)

- (LNKVector)computeResiduals
{
	LNKMatrix *const matrix = self.matrix;
	const LNKSize rowCount = matrix.rowCount;
	const LNKSize columnCount = matrix.columnCount;

	LNKFloat *predictions = LNKFloatAlloc(rowCount);
	LNK_mmul(matrix.matrixBuffer, UNIT_STRIDE, [self _thetaVector], UNIT_STRIDE, predictions, UNIT_STRIDE, rowCount, 1, columnCount);
	LNK_vsub(predictions, UNIT_STRIDE, matrix.outputVector, UNIT_STRIDE, predictions, UNIT_STRIDE, rowCount);

	return LNKVectorCreateUnsafe(predictions, rowCount);
}

- (LNKVector)computeStandardizedResiduals
{
	LNKMatrix *hatMatrix = [self.hatMatrix retain];
	LNKVector residuals = [self computeResiduals];
	LNKFloat dot = 0;
	LNK_dotpr(residuals.data, UNIT_STRIDE, residuals.data, UNIT_STRIDE, &dot, residuals.length);

	const LNKSize rowCount = residuals.length;
	const LNKFloat normalizer = dot / rowCount;

	LNKFloat *standardizedResiduals = LNKFloatAlloc(rowCount);

	for (LNKSize row = 0; row < rowCount; row++) {
		const LNKFloat variance = normalizer * (1 - [hatMatrix valueAtRow:row column:row]);
		standardizedResiduals[row] = residuals.data[row] / LNK_sqrt(variance);
	}

	[hatMatrix release];
	LNKVectorRelease(residuals);

	return LNKVectorCreateUnsafe(standardizedResiduals, rowCount);
}

- (LNKFloat)computeAIC
{
	LNKMatrix *const matrix = self.matrix;
	const LNKSize rowCount = matrix.rowCount;

	LNKVector residuals = [self computeResiduals];
	LNKFloat residualSum = 0;
	LNK_dotpr(residuals.data, UNIT_STRIDE, residuals.data, UNIT_STRIDE, &residualSum, rowCount);
	LNKVectorRelease(residuals);

	const LNKFloat term1 = rowCount * (LNKLog(2 * M_PI) + 1 + LNKLog(residualSum / rowCount));
	const LNKSize k = matrix.columnCount + 1;
	const LNKFloat term2 = (LNKFloat)k * 2;
	return term1 + term2;
}

- (LNKFloat)computeBIC
{
	LNKMatrix *const matrix = self.matrix;
	const LNKSize rowCount = matrix.rowCount;

	LNKVector residuals = [self computeResiduals];
	LNKFloat residualSum = 0;
	LNK_dotpr(residuals.data, UNIT_STRIDE, residuals.data, UNIT_STRIDE, &residualSum, rowCount);
	LNKVectorRelease(residuals);

	const LNKFloat term1 = rowCount * (LNKLog(2 * M_PI) + 1 + LNKLog(residualSum / rowCount));
	const LNKSize k = matrix.columnCount + 1;
	const LNKFloat term2 = (LNKFloat)k * LNKLog(rowCount);
	return term1 + term2;
}

- (LNKFloat)computeR2
{
	LNKMatrix *const matrix = self.matrix;
	const LNKFloat *outputVector = matrix.outputVector;
	const LNKSize rowCount = matrix.rowCount;

	LNKVector residuals = [self computeResiduals];
	LNKFloat ssRes = 0;
	LNK_dotpr(residuals.data, UNIT_STRIDE, residuals.data, UNIT_STRIDE, &ssRes, residuals.length);
	LNKVectorRelease(residuals);

	LNKFloat ssTot = 0;

	LNKFloat mean = 0;
	LNK_vmean(outputVector, UNIT_STRIDE, &mean, rowCount);

	const LNKFloat minusMean = -mean;
	LNKFloat *const meanAveragedOutput = LNKFloatAlloc(rowCount);
	LNK_vsadd(outputVector, UNIT_STRIDE, &minusMean, meanAveragedOutput, UNIT_STRIDE, rowCount);
	LNK_dotpr(meanAveragedOutput, UNIT_STRIDE, meanAveragedOutput, UNIT_STRIDE, &ssTot, rowCount);
	free(meanAveragedOutput);

	return 1 - ssRes / ssTot;
}

- (LNKMatrix *)hatMatrix
{
	LNKMatrix *const matrix = self.matrix;
	const LNKSize rowCount = matrix.rowCount;
	const LNKSize columnCount = matrix.columnCount;
	const LNKFloat *const matrixBuffer = matrix.matrixBuffer;

	LNKFloat *const transpose = LNKFloatAlloc(rowCount * columnCount);
	LNK_mtrans(matrixBuffer, transpose, columnCount, rowCount);

	LNKFloat *const square = LNKFloatAlloc(columnCount * columnCount);
	LNK_mmul(transpose, UNIT_STRIDE, matrixBuffer, UNIT_STRIDE, square, UNIT_STRIDE, columnCount, columnCount, rowCount);

	LNK_minvert(square, columnCount);

	LNKFloat *const workspace = LNKFloatAlloc(rowCount * columnCount);
	LNK_mmul(square, UNIT_STRIDE, transpose, UNIT_STRIDE, workspace, UNIT_STRIDE, columnCount, rowCount, columnCount);


	LNKMatrix *const hatMatrix = [[LNKMatrix alloc] initWithRowCount:rowCount columnCount:rowCount prepareBuffers:^BOOL(LNKFloat *matrixData, LNKFloat *outputVector) {
#pragma unused(outputVector)
		LNK_mmul(matrixBuffer, UNIT_STRIDE, workspace, UNIT_STRIDE, matrixData, UNIT_STRIDE, rowCount, rowCount, columnCount);
		return YES;
	}];

	free(transpose);
	free(square);
	free(workspace);

	return [hatMatrix autorelease];
}

@end
