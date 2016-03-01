//
//  _LNKLinRegPredictorNE_AC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKLinRegPredictorNE_AC.h"

#import "LNKAccelerate.h"
#import "LNKLinRegPredictorPrivate.h"
#import "LNKMatrix.h"
#import "LNKPredictorPrivate.h"
#import "LNKUtilities.h"

@implementation _LNKLinRegPredictorNE_AC

- (void)train {
	LNKMatrix *matrix = self.matrix;
	const LNKSize rowCount = matrix.rowCount;
	const LNKSize columnCount = matrix.columnCount;
	const LNKFloat *matrixBuffer = matrix.matrixBuffer;
	
	LNKFloat *transpose = LNKFloatAlloc(rowCount * columnCount);
	LNK_mtrans(matrixBuffer, transpose, columnCount, rowCount);
	
	LNKFloat *square = LNKFloatAlloc(columnCount * columnCount);
	LNK_mmul(transpose, UNIT_STRIDE, matrixBuffer, UNIT_STRIDE, square, UNIT_STRIDE, columnCount, columnCount, rowCount);
	
	LNK_minvert(square, columnCount);
	
	LNKFloat *workspace = LNKFloatAlloc(rowCount * columnCount);
	LNK_mmul(square, UNIT_STRIDE, transpose, UNIT_STRIDE, workspace, UNIT_STRIDE, columnCount, rowCount, columnCount);
	LNK_mmul(workspace, UNIT_STRIDE, matrix.outputVector, UNIT_STRIDE, [self _thetaVector], UNIT_STRIDE, columnCount, 1, rowCount);
	
	free(transpose);
	free(square);
	free(workspace);
}

@end
