//
//  _LNKLinRegPredictorNE_AC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKLinRegPredictorNE_AC.h"

#import "LNKAccelerate.h"
#import "LNKMatrix.h"
#import "LNKLinRegPredictorPrivate.h"
#import "LNKPredictorPrivate.h"
#import "LNKUtilities.h"

@implementation _LNKLinRegPredictorNE_AC

- (void)train {
	LNKMatrix *matrix = self.matrix;
	const LNKSize exampleCount = matrix.exampleCount;
	const LNKSize columnCount = matrix.columnCount;
	const LNKFloat *matrixBuffer = matrix.matrixBuffer;
	
	LNKFloat *transpose = LNKFloatAlloc(exampleCount * columnCount);
	LNK_mtrans(matrixBuffer, UNIT_STRIDE, transpose, UNIT_STRIDE, columnCount, exampleCount);
	
	LNKFloat *square = LNKFloatAlloc(columnCount * columnCount);
	LNK_mmul(transpose, UNIT_STRIDE, matrixBuffer, UNIT_STRIDE, square, UNIT_STRIDE, columnCount, columnCount, exampleCount);
	
	LNK_minvert(square, columnCount);
	
	LNKFloat *workspace = LNKFloatAlloc(exampleCount * columnCount);
	LNK_mmul(square, UNIT_STRIDE, transpose, UNIT_STRIDE, workspace, UNIT_STRIDE, columnCount, exampleCount, columnCount);
	LNK_mmul(workspace, UNIT_STRIDE, matrix.outputVector, UNIT_STRIDE, [self _thetaVector], UNIT_STRIDE, columnCount, 1, exampleCount);
	
	free(transpose);
	free(square);
	free(workspace);
}

@end
