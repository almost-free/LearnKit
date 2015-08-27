//
//  LNKUtilities.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKUtilities.h"

void LNKPrintVector(const char *name, LNKFloat *vector, LNKSize n) {
	NSCAssert(name, @"The name must not be NULL");
	NSCAssert(vector, @"The vector must not be NULL");
	
	printf("Vector '%s':\n", name);
	for (LNKSize i = 0; i < n; i++) {
		printf("%g\n", vector[i]);
	}
	printf("End of Vector '%s'\n\n", name);
}

void LNKPrintMatrix(const char *name, LNKFloat *matrix, LNKSize m, LNKSize n) {
	NSCAssert(name, @"The name must not be NULL");
	NSCAssert(matrix, @"The matrix must not be NULL");
	
	printf("Matrix '%s':\n", name);
	for (LNKSize i = 0; i < m; i++) {
		for (LNKSize j = 0; j < n; j++) {
			printf("%g\t", matrix[i * n + j]);
		}
		printf("\n");
	}
	printf("End of Matrix '%s'\n\n", name);
}

NSData *LNKLoadBinaryMatrixFromFileAtURL(NSURL *url, LNKSize expectedLength) {
	NSCAssert(url, @"The url must not be nil");
	NSCAssert(expectedLength, @"The expected length must be greater than 0");
	
	NSError *error = nil;
	NSData *matrixData = [NSData dataWithContentsOfURL:url options:0 error:&error];

	if (!matrixData) {
		NSLog(@"Could not load the binary matrix file (%@): %@", url, error);
		return nil;
	}

	if (matrixData.length != expectedLength) {
		NSLog(@"The size of the binary matrix file (%@) is invalid: %@", url, error);
		return nil;
	}
	
	return matrixData;
}
