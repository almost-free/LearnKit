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

	const LNKSize rowCount = workingMatrix.rowCount;
	const LNKFloat *matrixBuffer = workingMatrix.matrixBuffer;

	LNKFloat *const sigmaMatrix = LNKFloatAlloc(columnCount * columnCount);

	// S = 1/m X' X
	// After normalization, this holds the covariance matrix.
	LNKFloat *const transposeMatrix = LNKFloatAlloc(columnCount * rowCount);
	LNK_mtrans(matrixBuffer, transposeMatrix, columnCount, rowCount);

	LNK_mmul(transposeMatrix, UNIT_STRIDE, matrixBuffer, UNIT_STRIDE, sigmaMatrix, UNIT_STRIDE, columnCount, columnCount, rowCount);
	free(transposeMatrix);

	const LNKFloat m = (LNKFloat)rowCount;
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

	LNKMatrix *const rotationMatrix = [[LNKMatrix alloc] initWithRowCount:columnCount columnCount:columnCount addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *localMatrix, LNKFloat *outputVector) {
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

- (LNKPCAInformation *)analyzeApproximatePrincipalComponents:(LNKSize)principalComponents toTolerance:(LNKFloat)tolerance maximumIterations:(LNKSize)maximumIterations {
	if (tolerance == LNKFloatMax && maximumIterations == LNKSizeMax) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"tolerance and maximumIterations can't both be maximum or the algorithm will never converge" userInfo:nil];
	}

	LNKMatrix *workingMatrix = nil;

	if (self.normalized) {
		workingMatrix = [self retain];
	} else {
		workingMatrix = [self.normalizedMatrix retain];
	}

	const LNKSize rowCount = self.rowCount;
	const LNKSize columnCount = self.columnCount;

	NSAssert(rowCount == workingMatrix.rowCount, @"These dimensions should match");
	NSAssert(columnCount == workingMatrix.columnCount, @"These dimensions should match");

	LNKFloat *const workingMatrixBuffer = LNKFloatAllocAndCopy(workingMatrix.matrixBuffer, columnCount * rowCount);
	LNKFloat *const components = LNKFloatCalloc(columnCount * columnCount);

	// Re-used vectors across iterations.
	LNKFloat *const matrixTranspose = LNKFloatAlloc(columnCount * rowCount);
	LNKFloat *const shadowMatrix = LNKFloatAlloc(columnCount * rowCount);
	LNKFloat *const u = LNKFloatAlloc(columnCount);
	LNKFloat *const wNext = LNKFloatAlloc(rowCount);
	LNKFloat *const uNext = LNKFloatAlloc(columnCount);
	LNKFloat *const errorVector = LNKFloatAlloc(columnCount);

	void(^findPrincipalComponent)(LNKSize) = ^(LNKSize index) {
		LNK_mtrans(workingMatrixBuffer, matrixTranspose, columnCount, rowCount);

		// Start with the first row.
		LNKFloatCopy(u, workingMatrixBuffer, columnCount);

		for (LNKSize iteration = 0; iteration < maximumIterations; iteration++) {
			// w_next = (featureMatrix * u) / (t(u) * u)
			LNK_mmul(workingMatrixBuffer, UNIT_STRIDE, u, UNIT_STRIDE, wNext, UNIT_STRIDE, rowCount, 1, columnCount);

			LNKFloat uNorm = 0;
			LNK_dotpr(u, UNIT_STRIDE, u, UNIT_STRIDE, &uNorm, columnCount);
			LNK_vsdiv(wNext, UNIT_STRIDE, &uNorm, wNext, UNIT_STRIDE, rowCount);

			// u_next = (t(featureMatrix) * w_next) / (t(w_next) * w_next)
			LNKFloat wNorm = 0;
			LNK_dotpr(wNext, UNIT_STRIDE, wNext, UNIT_STRIDE, &wNorm, rowCount);
			LNK_mmul(matrixTranspose, UNIT_STRIDE, wNext, UNIT_STRIDE, uNext, UNIT_STRIDE, columnCount, 1, rowCount);
			LNK_vsdiv(uNext, UNIT_STRIDE, &wNorm, uNext, UNIT_STRIDE, columnCount);

			// s = sqrt(t(u_next) * u_next)
			LNKFloat s = 0;
			LNK_dotpr(uNext, UNIT_STRIDE, uNext, UNIT_STRIDE, &s, columnCount);
			s = LNK_sqrt(s);

			// u_next = u_next / s
			// w_next = s * w_next
			LNK_vsdiv(uNext, UNIT_STRIDE, &s, uNext, UNIT_STRIDE, columnCount);
			LNK_vsmul(wNext, UNIT_STRIDE, &s, wNext, UNIT_STRIDE, rowCount);

			// error = u - u_next
			LNK_vsub(uNext, UNIT_STRIDE, u, UNIT_STRIDE, errorVector, UNIT_STRIDE, columnCount);

			LNKFloat errorNorm = 0;
			LNK_dotpr(errorVector, UNIT_STRIDE, errorVector, UNIT_STRIDE, &errorNorm, columnCount);

			LNKFloatCopy(u, uNext, columnCount);

			if (LNK_sqrt(errorNorm) < tolerance) {
				break;
			}
		}

		LNK_mmov(u, components + index, 1, columnCount, 1, columnCount);

		// X = X - w * t(u)
		LNK_mmul(wNext, UNIT_STRIDE, u, UNIT_STRIDE, shadowMatrix, UNIT_STRIDE, rowCount, columnCount, 1);
		LNK_vsub(shadowMatrix, UNIT_STRIDE, workingMatrixBuffer, UNIT_STRIDE, workingMatrixBuffer, UNIT_STRIDE, columnCount * rowCount);
	};

	for (LNKSize componentIndex = 0; componentIndex < MIN(principalComponents, columnCount); componentIndex++) {
		findPrincipalComponent(componentIndex);
	}

	free(matrixTranspose);
	free(shadowMatrix);
	free(u);
	free(wNext);
	free(uNext);
	free(errorVector);

	const LNKVector centers = LNKVectorAllocAndCopy(workingMatrix.normalizationMeanVector, columnCount);
	const LNKVector scales = LNKVectorAllocAndCopy(workingMatrix.normalizationStandardDeviationVector, columnCount);
	const LNKFloat *const standardDeviations = LNKFloatCalloc(columnCount);

	LNKMatrix *const rotationMatrix = [[LNKMatrix alloc] initWithRowCount:columnCount columnCount:columnCount addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
#pragma unused(outputVector)
		LNKFloatCopy(matrix, components, columnCount * columnCount);
		return YES;
	}];
	LNKMatrix *const rotatedMatrix = [workingMatrix multiplyByMatrix:rotationMatrix];

	LNKPCAInformation *const pca = [[LNKPCAInformation alloc] initWithCenters:centers scales:scales rotationMatrix:rotationMatrix standardDeviations:LNKVectorMakeUnsafe(standardDeviations, columnCount) rotatedMatrix:rotatedMatrix];
	[rotationMatrix release];
	[workingMatrix release];

	return [pca autorelease];
}

- (LNKPCAInformation *)analyzeApproximatePrincipalComponents:(LNKSize)principalComponents {
	return [self analyzeApproximatePrincipalComponents:principalComponents toTolerance:1e-9 maximumIterations:500];
}

- (LNKMatrix *)matrixReducedToDimension:(LNKSize)dimension {
	if (dimension < 1) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"The dimension must be >= 1" userInfo:nil];
	}
	
	const LNKSize columnCount = self.columnCount;

	if (dimension >= columnCount) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"The dimension must be less than the column count" userInfo:nil];
	}

	const LNKSize rowCount = self.rowCount;

	LNKPCAInformation *const pca = [[self analyzePrincipalComponents] retain];

	if (pca == nil) {
		return nil;
	}

	LNKFloat *const relevantMatrix = LNKFloatAlloc(dimension * rowCount);
	LNK_mmov(pca.rotatedMatrix.matrixBuffer, relevantMatrix, dimension, rowCount, columnCount, dimension);

	LNKMatrix *const resultingMatrix = [[LNKMatrix alloc] initWithRowCount:rowCount columnCount:dimension addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
		LNKFloatCopy(matrix, relevantMatrix, rowCount * dimension);
		LNKFloatCopy(outputVector, self.outputVector, rowCount);
		return YES;
	}];

	[pca release];
	free(relevantMatrix);

	return [resultingMatrix autorelease];
}

- (LNKMatrix *)matrixProjectedToDimension:(LNKSize)dimension withPCAInformation:(LNKPCAInformation *)pca {
	if (pca == nil) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"PCA information must be passed in" userInfo:nil];
	}

	if (dimension < 1) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"The dimension must be >= 1" userInfo:nil];
	}

	const LNKSize columnCount = self.columnCount;

	if (dimension >= columnCount) {
		@throw [NSException exceptionWithName:NSInvalidArgumentException reason:@"The dimension must be less than the column count" userInfo:nil];
	}

	const LNKSize rowCount = self.rowCount;

	LNKMatrix *const rotationMatrix = pca.rotationMatrix;
	const LNKSize rotationMatrixRows = rotationMatrix.rowCount;
	const LNKSize rotationMatrixColumns = rotationMatrix.columnCount;

	// Zeros out the eigenvectors corresponding to the irrelevant principal components.
	LNKMatrix *const clearedMatrix = [[LNKMatrix alloc] initWithRowCount:rotationMatrixRows columnCount:rotationMatrixColumns addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *localMatrix, LNKFloat *outputVector) {
	#pragma unused(outputVector)
		LNKFloatCopy(localMatrix, rotationMatrix.matrixBuffer, rotationMatrixRows * rotationMatrixColumns);

		for (LNKSize d = dimension; d < pca.rotationMatrix.columnCount; d++) {
			for (LNKSize m = 0; m < pca.rotationMatrix.rowCount; m++) {
				localMatrix[m * pca.rotationMatrix.columnCount + d] = 0;
			}
		}

		return YES;
	}];

	LNKMatrix *const nextRotationMatrix = clearedMatrix.transposedMatrix;
	[clearedMatrix release];

	LNKFloat *const result = LNKFloatAlloc(rowCount * columnCount);
	LNK_mmul(pca.rotatedMatrix.matrixBuffer, UNIT_STRIDE, nextRotationMatrix.matrixBuffer, UNIT_STRIDE, result, UNIT_STRIDE, rowCount, columnCount, columnCount);

	const LNKFloat *const centers = pca.centers.data;
	const LNKFloat *const scales = pca.scales.data;

	for (LNKSize column = 0; column < columnCount; column++) {
		LNKFloat *const columnPointer = result + column;

		const LNKFloat mean = centers[column];
		const LNKFloat sd = scales[column];

		LNK_vsmul(columnPointer, columnCount, &sd, columnPointer, columnCount, rowCount);
		LNK_vsadd(columnPointer, columnCount, &mean, columnPointer, columnCount, rowCount);
	}

	LNKMatrix *const resultingMatrix = [[LNKMatrix alloc] initWithRowCount:rowCount columnCount:columnCount addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
		LNKFloatCopy(matrix, result, rowCount * columnCount);
		LNKFloatCopy(outputVector, self.outputVector, rowCount);
		return YES;
	}];

	free(result);

	return [resultingMatrix autorelease];
}

@end
