//
//  LNKAccelerate.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKAccelerate.h"

void LNK_mtrans(const LNKFloat *source, LNKFloat *dest, vDSP_Length N, vDSP_Length M) {
	if (source == dest) {
#if USE_DOUBLE_PRECISION
		vDSP_mtransD(source, UNIT_STRIDE, dest, UNIT_STRIDE, N, M);
#else
		vDSP_mtrans(source, UNIT_STRIDE, dest, UNIT_STRIDE, N, M);
#endif
		return;
	}

#define BLOCK_SIZE 16
	
	for (vDSP_Length i = 0; i < N; i += BLOCK_SIZE) {
		for (vDSP_Length j = 0; j < M; j += BLOCK_SIZE) {
			const vDSP_Length imax = MIN(i + BLOCK_SIZE, N);
			const vDSP_Length jmax = MIN(j + BLOCK_SIZE, M);
			
			for (vDSP_Length k = i; k < imax; ++k) {
				for (vDSP_Length l = j; l < jmax; ++l) {
					dest[l + k*M] = source[k + l*N];
				}
			}
		}
	}
}

void LNK_minvert(LNKFloat *matrix, LNKSize n) {
	NSCAssert(matrix, @"The matrix must not be NULL");
	NSCAssert(n, @"The length must be greater than 0");
	
	__CLPK_integer *pivot = malloc(n * n * sizeof(__CLPK_integer));
	__CLPK_integer error = 0;
	__CLPK_integer np = (__CLPK_integer)n;
	LNKFloat *workspace = LNKFloatAlloc(n);
	
#if USE_DOUBLE_PRECISION
	dgetrf_(&np, &np, matrix, &np, pivot, &error);
	dgetri_(&np, matrix, &np, pivot, workspace, &np, &error);
#else
	sgetrf_(&np, &np, matrix, &np, pivot, &error);
	sgetri_(&np, matrix, &np, pivot, workspace, &np, &error);
#endif
	
	free(workspace);
	free(pivot);
}

void LNK_vsigmoid(LNKFloat *vector, LNKSize n) {
	NSCAssert(vector, @"The vector must not be NULL");
	NSCAssert(n, @"The length must be greater than 0");
	
	const int np = (int)n;
	const LNKFloat one = 1;
	
	// 1 / (1 + e^(-vector))
	LNK_vneg(vector, UNIT_STRIDE, vector, UNIT_STRIDE, n);
	LNK_vexp(vector, vector, &np);
	LNK_vsadd(vector, UNIT_STRIDE, &one, vector, UNIT_STRIDE, n);
	LNK_svdiv(&one, vector, UNIT_STRIDE, vector, UNIT_STRIDE, n);
}

void LNK_vsigmoidgrad(const LNKFloat *vector, LNKFloat *outVector, LNKSize n) {
	NSCAssert(vector, @"The vector must not be NULL");
	NSCAssert(outVector, @"The out vector must not be NULL");
	NSCAssert(n, @"The length must be greater than 0");
	
	// vector (1 - vector) = vector - vector^2
	LNKFloat *vectorSquared = LNKFloatAlloc(n);
	LNK_vsq(vector, UNIT_STRIDE, vectorSquared, UNIT_STRIDE, n);
	
	LNK_vsub(vectorSquared, UNIT_STRIDE, vector, UNIT_STRIDE, outVector, UNIT_STRIDE, n);
	free(vectorSquared);
}

LNKFloat LNK_vsd(LNKVector vector, LNKSize stride, LNKFloat *workgroup, LNKFloat mean, BOOL inSample) {
	NSCAssert(vector.data != NULL, @"The vector must not be NULL");
	NSCAssert(vector.length > 0, @"The length must be greater than 0");

	BOOL shouldFreeWorkgroup = NO;
	if (workgroup == NULL) {
		workgroup = LNKFloatAlloc(vector.length);
		shouldFreeWorkgroup = YES;
	}
	
	const LNKFloat minusMean = -mean;
	LNK_vsadd(vector.data, stride, &minusMean, workgroup, UNIT_STRIDE, vector.length);

	LNKFloat sd;
	LNK_dotpr(workgroup, UNIT_STRIDE, workgroup, UNIT_STRIDE, &sd, vector.length);

	if (shouldFreeWorkgroup) {
		free(workgroup);
	}

	const LNKSize adjustedN = inSample ? vector.length - 1 : vector.length;
	return LNK_sqrt(1.0 / adjustedN * sd);
}

LNKFloat LNK_mdet(const LNKFloat *matrix, LNKSize n) {
	NSCAssert(matrix, @"The matrix must not be NULL");
	NSCAssert(n, @"The length must be greater than 0");
	
	__CLPK_integer np = (__CLPK_integer)n;
	__CLPK_integer error = 0;
	__CLPK_integer *pivot = malloc(n * sizeof(__CLPK_integer));
	
	LNKFloat *matrixCopy = LNKFloatAllocAndCopy(matrix, n * n);
	
#if USE_DOUBLE_PRECISION
	dgetrf_(&np, &np, matrixCopy, &np, pivot, &error);
#else
	sgetrf_(&np, &np, matrixCopy, &np, pivot, &error);
#endif
	
	if (error > 0) {
		fprintf(stderr, "LNK_mdet: we have a singular matrix\n");
		free(matrixCopy);
		free(pivot);
		return 0;
	}
	
	LNKFloat determinant = 1;
	
	// Multiply the diagonal elements.
	for (__CLPK_integer index = 0; index < np; index++) {
		determinant *= matrixCopy[index * n + index];
		
		if (pivot[index] != (index+1))
			determinant *= -1;
	}
	
	free(matrixCopy);
	free(pivot);
	
	return determinant;
}
