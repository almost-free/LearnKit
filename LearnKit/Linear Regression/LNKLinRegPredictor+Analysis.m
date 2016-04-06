//
//  LNKLinRegPredictor+Analysis.m
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKLinRegPredictor+Analysis.h"

#import "LNKAccelerate.h"
#import "LNKLinRegPredictorPrivate.h"
#import "LNKMatrix.h"

@implementation LNKLinRegPredictor (Analysis)

- (LNKVector)computeResiduals
{
	LNKMatrix *const matrix = self.matrix;
	const LNKSize rowCount = matrix.rowCount;
	const LNKSize columnCount = matrix.columnCount;

	LNKFloat *predictions = LNKFloatAlloc(rowCount);
	LNK_mmul(matrix.matrixBuffer, UNIT_STRIDE, [self _thetaVector], UNIT_STRIDE, predictions, UNIT_STRIDE, rowCount, 1, columnCount);
	LNK_vsub(matrix.outputVector, UNIT_STRIDE, predictions, UNIT_STRIDE, predictions, UNIT_STRIDE, rowCount);

	return LNKVectorMakeUnsafe(predictions, rowCount);
}

- (LNKFloat)computeAIC
{
	LNKMatrix *const matrix = self.matrix;
	const LNKSize rowCount = matrix.rowCount;

	LNKVector residuals = [self computeResiduals];
	LNKFloat residualSum = 0;
	LNK_dotpr(residuals.data, UNIT_STRIDE, residuals.data, UNIT_STRIDE, &residualSum, rowCount);
	LNKVectorFree(residuals);

	const LNKFloat term1 = rowCount * (LNKLog(2 * M_PI) + 1 + LNKLog(residualSum / rowCount));
	const LNKSize k = matrix.columnCount + 1;
	const LNKFloat term2 = (LNKFloat)k * 2;
	return term1 + term2;
}

@end
