//
//  LNKAccelerate.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKAccelerate.h"

void LNK_mtrans(const LNKFloat *source, LNKFloat *dest, vDSP_Length N, vDSP_Length M) {
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
	assert(matrix);
	assert(n);
	
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
	assert(vector);
	assert(n);
	
	const int np = (int)n;
	const LNKFloat one = 1;
	
	// 1 / (1 + e^(-vector))
	LNK_vneg(vector, UNIT_STRIDE, vector, UNIT_STRIDE, n);
	LNK_vexp(vector, vector, &np);
	LNK_vsadd(vector, UNIT_STRIDE, &one, vector, UNIT_STRIDE, n);
	LNK_svdiv(&one, vector, UNIT_STRIDE, vector, UNIT_STRIDE, n);
}

void LNK_vsigmoidgrad(const LNKFloat *vector, LNKFloat *outVector, LNKSize n) {
	assert(vector);
	assert(outVector);
	assert(n);
	
	// vector (1 - vector) = vector - vector^2
	LNKFloat *vectorSquared = LNKFloatAlloc(n);
	LNK_vsq(vector, UNIT_STRIDE, vectorSquared, UNIT_STRIDE, n);
	
	LNK_vsub(vectorSquared, UNIT_STRIDE, vector, UNIT_STRIDE, outVector, UNIT_STRIDE, n);
	free(vectorSquared);
}

LNKFloat LNK_vsd(const LNKFloat *vector, LNKSize n, LNKSize stride, LNKFloat *workgroup, LNKFloat mean, BOOL inSample) {
	assert(vector);
	assert(workgroup);
	assert(n);
	
	const LNKFloat minusMean = -mean;
	LNK_vsadd(vector, stride, &minusMean, workgroup, UNIT_STRIDE, n);

	LNKFloat sd;
	LNK_dotpr(workgroup, UNIT_STRIDE, workgroup, UNIT_STRIDE, &sd, n);
	
	const LNKSize adjustedN = inSample ? n - 1 : n;
	
	return LNK_sqrt(1.0 / adjustedN * sd);
}

LNKFloat LNK_mdet(const LNKFloat *matrix, LNKSize n) {
	assert(matrix);
	assert(n);
	
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
