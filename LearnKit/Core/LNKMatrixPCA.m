//
//  LNKMatrixPCA.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKMatrixPCA.h"

#import "LNKAccelerate.h"

@implementation LNKMatrix (PCA)

- (LNKMatrix *)matrixReducedToDimension:(LNKSize)dimension {
	if (dimension < 1) {
		@throw [NSException exceptionWithName:NSInternalInconsistencyException reason:@"The dimension must be >= 1" userInfo:nil];
	}
	
	const LNKSize columnCount = self.columnCount;
	
	if (dimension >= columnCount) {
		@throw [NSException exceptionWithName:NSInternalInconsistencyException reason:@"The dimension must be less than the column count" userInfo:nil];
	}
	
	if (!self.isNormalized) {
		@throw [NSException exceptionWithName:NSInternalInconsistencyException reason:@"Matrices must be normalized before being reduced" userInfo:nil];
	}
	
	const LNKSize exampleCount = self.exampleCount;
	const LNKFloat *matrixBuffer = self.matrixBuffer;
	
	LNKFloat *sigmaMatrix = LNKFloatAlloc(columnCount * columnCount);
	
	// S = 1/m X' X
	LNKFloat *transposeMatrix = LNKFloatAlloc(columnCount * exampleCount);
	LNK_mtrans(matrixBuffer, UNIT_STRIDE, transposeMatrix, UNIT_STRIDE, columnCount, exampleCount);
	
	LNK_mmul(transposeMatrix, UNIT_STRIDE, matrixBuffer, UNIT_STRIDE, sigmaMatrix, UNIT_STRIDE, columnCount, columnCount, exampleCount);
	free(transposeMatrix);
	
	const LNKFloat m = (LNKFloat)exampleCount;
	LNK_vsdiv(sigmaMatrix, UNIT_STRIDE, &m, sigmaMatrix, UNIT_STRIDE, columnCount * columnCount);
	
	__CLPK_integer columnCountCLPK = (__CLPK_integer)columnCount;
	__CLPK_integer info = 0;
	__CLPK_integer workSize = columnCountCLPK * 5;
	
	LNKFloat *s = LNKFloatAlloc(columnCountCLPK * columnCountCLPK);
	LNKFloat *u = LNKFloatAlloc(columnCountCLPK * columnCountCLPK);
	LNKFloat *work = LNKFloatAlloc(workSize);
	
	// Decompose S into U.
	int result = dgesvd_("A", "N", &columnCountCLPK, &columnCountCLPK, sigmaMatrix, &columnCountCLPK, s, u, &columnCountCLPK, NULL, &columnCountCLPK, work, &workSize, &info);
	free(sigmaMatrix);
	free(s);
	free(work);
	
	if (result != 0) {
		NSLog(@"%s: Singular value decomposition could not be performed (%d)", __PRETTY_FUNCTION__, result);
		free(u);
		return nil;
	}
	
	LNKFloat *relevantU = LNKFloatAlloc(dimension * columnCountCLPK);
	LNK_mmov(u, relevantU, dimension, columnCountCLPK, columnCountCLPK, dimension);
	free(u);
	
	// Project the matrix along U(:, 1:dimension).
	LNKFloat *z = LNKFloatAlloc(exampleCount * dimension);
	LNK_mmul(matrixBuffer, UNIT_STRIDE, relevantU, UNIT_STRIDE, z, UNIT_STRIDE, exampleCount, dimension, columnCountCLPK);
	free(relevantU);
	
	return [[[LNKMatrix alloc] initWithExampleCount:exampleCount columnCount:dimension addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
		LNKFloatCopy(matrix, z, exampleCount * dimension);
		LNKFloatCopy(outputVector, self.outputVector, exampleCount);
		return YES;
	}] autorelease];
}

@end
