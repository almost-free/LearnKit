//
//  _LNKLinRegPredictorNE_AC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKLinRegPredictorNE_AC.h"

#import "LNKAccelerate.h"
#import "LNKDesignMatrix.h"
#import "LNKLinRegPredictorPrivate.h"
#import "LNKPredictorPrivate.h"
#import "LNKUtilities.h"

@implementation _LNKLinRegPredictorNE_AC

- (void)train {
	LNKDesignMatrix *designMatrix = self.designMatrix;
	const LNKSize exampleCount = designMatrix.exampleCount;
	const LNKSize columnCount = designMatrix.columnCount;
	const LNKFloat *matrixBuffer = designMatrix.matrixBuffer;
	
	LNKFloat *transpose = LNKFloatAlloc(exampleCount * columnCount);
	LNK_mtrans(matrixBuffer, UNIT_STRIDE, transpose, UNIT_STRIDE, columnCount, exampleCount);
	
	LNKFloat *square = LNKFloatAlloc(columnCount * columnCount);
	LNK_mmul(transpose, UNIT_STRIDE, matrixBuffer, UNIT_STRIDE, square, UNIT_STRIDE, columnCount, columnCount, exampleCount);
	
	LNK_minvert(square, columnCount);
	
	LNKFloat *workspace = LNKFloatAlloc(exampleCount * columnCount);
	LNK_mmul(square, UNIT_STRIDE, transpose, UNIT_STRIDE, workspace, UNIT_STRIDE, columnCount, exampleCount, columnCount);
	LNK_mmul(workspace, UNIT_STRIDE, designMatrix.outputVector, UNIT_STRIDE, [self _thetaVector], UNIT_STRIDE, columnCount, 1, exampleCount);
	
	free(transpose);
	free(square);
	free(workspace);
}

@end
