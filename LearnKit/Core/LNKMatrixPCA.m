//
//  LNKMatrixPCA.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKMatrixPCA.h"

#import "LNKAccelerate.h"

@implementation LNKPCAInformation

// Ownership of LNKVectors is transferred.
- (instancetype)initWithCenters:(LNKVector)centers scales:(LNKVector)scales rotationMatrix:(LNKMatrix *)rotationMatrix standardDeviations:(LNKVector)standardDeviations rotatedMatrix:(LNKMatrix *)rotatedMatrix {
	if (!(self = [super init])) {
		return nil;
	}

	_centers = centers;
	_scales = scales;
	_standardDeviations = standardDeviations;
	_rotationMatrix = [rotationMatrix retain];
	_rotatedMatrix = [rotatedMatrix retain];

	return self;
}

- (void)dealloc {
	[_rotationMatrix release];
	[_rotatedMatrix release];
	LNKVectorFree(_centers);
	LNKVectorFree(_scales);
	LNKVectorFree(_standardDeviations);
	[super dealloc];
}

@end


@implementation LNKMatrix (PCA)

- (LNKPCAInformation *)analyzePrincipalComponents {
	const LNKSize columnCount = self.columnCount;

	LNKMatrix *workingMatrix = nil;

	if (self.normalized) {
		workingMatrix = [self retain];
	} else {
		workingMatrix = [self.normalizedMatrix retain];
	}

	NSAssert(columnCount == workingMatrix.columnCount, @"The number of columns should not change");

	const LNKSize exampleCount = workingMatrix.exampleCount;
	const LNKFloat *matrixBuffer = workingMatrix.matrixBuffer;

	LNKFloat *const sigmaMatrix = LNKFloatAlloc(columnCount * columnCount);

	// S = 1/m X' X
	// After normalization, this holds the covariance matrix.
	LNKFloat *const transposeMatrix = LNKFloatAlloc(columnCount * exampleCount);
	LNK_mtrans(matrixBuffer, transposeMatrix, columnCount, exampleCount);

	LNK_mmul(transposeMatrix, UNIT_STRIDE, matrixBuffer, UNIT_STRIDE, sigmaMatrix, UNIT_STRIDE, columnCount, columnCount, exampleCount);
	free(transposeMatrix);

	const LNKFloat m = (LNKFloat)exampleCount;
	LNK_vsdiv(sigmaMatrix, UNIT_STRIDE, &m, sigmaMatrix, UNIT_STRIDE, columnCount * columnCount);

	__CLPK_integer columnCountCLPK = (__CLPK_integer)columnCount;
	__CLPK_integer info = 0;

	LNKFloat *const s = LNKFloatAlloc(columnCountCLPK * columnCountCLPK);
	LNKFloat *const vt = LNKFloatAlloc(columnCountCLPK * columnCountCLPK);

	// We'll compute this dynamically!
	__CLPK_integer workSize = -1;
	LNKFloat workOptimal = 0;

	// The first pass computes work sizes.
	dgesvd_("N", "A", &columnCountCLPK, &columnCountCLPK, sigmaMatrix, &columnCountCLPK, s, NULL, &columnCountCLPK, vt, &columnCountCLPK, &workOptimal, &workSize, &info);

	workSize = (__CLPK_integer)workOptimal;
	LNKFloat *const work = LNKFloatAlloc(workSize);

	// The second pass to decomposes the covariance matrix into eigenvectors and eigenvalues.
	const int result = dgesvd_("N", "A", &columnCountCLPK, &columnCountCLPK, sigmaMatrix, &columnCountCLPK, s, NULL, &columnCountCLPK, vt, &columnCountCLPK, work, &workSize, &info);
	free(work);
	free(sigmaMatrix);

	if (result != 0) {
		NSLog(@"%s: Singular value decomposition could not be performed (%d)", __PRETTY_FUNCTION__, result);
		free(s);
		free(vt);
		[workingMatrix release];
		return nil;
	}

	const LNKVector centers = LNKVectorAllocAndCopy(workingMatrix.normalizationMeanVector, columnCount);
	const LNKVector scales = LNKVectorAllocAndCopy(workingMatrix.normalizationStandardDeviationVector, columnCount);
	LNKFloat *const standardDeviations = LNKFloatAlloc(columnCount);

	for (LNKSize column = 0; column < columnCount; column++) {
		standardDeviations[column] = LNK_sqrt(s[column]);
	}

	LNKMatrix *const rotationMatrix = [[LNKMatrix alloc] initWithExampleCount:columnCount columnCount:columnCount addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *localMatrix, LNKFloat *outputVector) {
#pragma unused(outputVector)
		LNKFloatCopy(localMatrix, vt, columnCount * columnCount);
		return YES;
	}];

	LNKMatrix *const rotatedMatrix = [workingMatrix multiplyByMatrix:rotationMatrix];
	[workingMatrix release];
	free(s);
	free(vt);

	LNKPCAInformation *const information = [[LNKPCAInformation alloc] initWithCenters:centers scales:scales rotationMatrix:rotationMatrix standardDeviations:LNKVectorMakeUnsafe(standardDeviations, columnCount) rotatedMatrix:rotatedMatrix];
	[rotationMatrix release];

	return [information autorelease];
}

- (LNKMatrix *)matrixReducedToDimension:(LNKSize)dimension {
	if (dimension < 1) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"The dimension must be >= 1" userInfo:nil];
	}
	
	const LNKSize columnCount = self.columnCount;

	if (dimension >= columnCount) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"The dimension must be less than the column count" userInfo:nil];
	}

	const LNKSize exampleCount = self.exampleCount;

	LNKPCAInformation *const pca = [[self analyzePrincipalComponents] retain];

	if (pca == nil) {
		return nil;
	}

	LNKFloat *const relevantMatrix = LNKFloatAlloc(dimension * exampleCount);
	LNK_mmov(pca.rotatedMatrix.matrixBuffer, relevantMatrix, dimension, exampleCount, columnCount, dimension);

	LNKMatrix *const resultingMatrix = [[LNKMatrix alloc] initWithExampleCount:exampleCount columnCount:dimension addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
		LNKFloatCopy(matrix, relevantMatrix, exampleCount * dimension);
		LNKFloatCopy(outputVector, self.outputVector, exampleCount);
		return YES;
	}];

	[pca release];
	free(relevantMatrix);

	return [resultingMatrix autorelease];
}

- (LNKMatrix *)matrixProjectedToDimension:(LNKSize)dimension {
	if (dimension < 1) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"The dimension must be >= 1" userInfo:nil];
	}

	const LNKSize columnCount = self.columnCount;

	if (dimension >= columnCount) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"The dimension must be less than the column count" userInfo:nil];
	}

	const LNKSize exampleCount = self.exampleCount;

	LNKPCAInformation *const pca = [[self analyzePrincipalComponents] retain];

	if (pca == nil) {
		return nil;
	}

	LNKMatrix *const rotationMatrix = pca.rotationMatrix;
	const LNKSize rotationMatrixRows = rotationMatrix.exampleCount;
	const LNKSize rotationMatrixColumns = rotationMatrix.columnCount;

	// Zeros out the eigenvectors corresponding to the irrelevant principal components.
	LNKMatrix *const clearedMatrix = [[LNKMatrix alloc] initWithExampleCount:rotationMatrixRows columnCount:rotationMatrixColumns addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *localMatrix, LNKFloat *outputVector) {
	#pragma unused(outputVector)
		LNKFloatCopy(localMatrix, rotationMatrix.matrixBuffer, rotationMatrixRows * rotationMatrixColumns);

		for (LNKSize d = dimension; d < pca.rotationMatrix.columnCount; d++) {
			for (LNKSize m = 0; m < pca.rotationMatrix.exampleCount; m++) {
				localMatrix[m * pca.rotationMatrix.columnCount + d] = 0;
			}
		}

		return YES;
	}];

	LNKMatrix *const nextRotationMatrix = clearedMatrix.transposedMatrix;
	[clearedMatrix release];

	LNKFloat *const result = LNKFloatAlloc(exampleCount * columnCount);
	LNK_mmul(pca.rotatedMatrix.matrixBuffer, UNIT_STRIDE, nextRotationMatrix.matrixBuffer, UNIT_STRIDE, result, UNIT_STRIDE, exampleCount, columnCount, columnCount);

	const LNKFloat *const centers = pca.centers.data;
	const LNKFloat *const scales = pca.scales.data;

	for (LNKSize column = 0; column < columnCount; column++) {
		LNKFloat *const columnPointer = result + column;

		const LNKFloat mean = centers[column];
		const LNKFloat sd = scales[column];

		LNK_vsmul(columnPointer, columnCount, &sd, columnPointer, columnCount, exampleCount);
		LNK_vsadd(columnPointer, columnCount, &mean, columnPointer, columnCount, exampleCount);
	}

	LNKMatrix *const resultingMatrix = [[LNKMatrix alloc] initWithExampleCount:exampleCount columnCount:columnCount addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
		LNKFloatCopy(matrix, result, exampleCount * columnCount);
		LNKFloatCopy(outputVector, self.outputVector, exampleCount);
		return YES;
	}];

	free(result);
	[pca release];

	return [resultingMatrix autorelease];
}

@end
