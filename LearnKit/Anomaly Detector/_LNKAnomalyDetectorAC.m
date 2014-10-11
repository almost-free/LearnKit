//
//  _LNKAnomalyDetectorAC.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "_LNKAnomalyDetectorAC.h"

#import "LNKAccelerate.h"
#import "LNKMatrix.h"

@implementation _LNKAnomalyDetectorAC {
	LNKFloat *_mu;
	LNKFloat *_sigma2;
}

- (void)train {
	LNKMatrix *matrix = self.matrix;
	const LNKSize columnCount = matrix.columnCount;
	const LNKSize exampleCount = matrix.exampleCount;
	const LNKFloat *matrixBuffer = matrix.matrixBuffer;
	
	_mu = LNKFloatAlloc(columnCount);
	_sigma2 = LNKFloatCalloc(columnCount * columnCount);
	
	LNKFloat *workgroup = LNKFloatAlloc(exampleCount);
	
	// Compute Gaussian parameters.
	for (LNKSize column = 0; column < columnCount; column++) {
		const LNKFloat *columnPointer = matrixBuffer + column;
		
		LNKFloat mean;
		LNK_vmean(columnPointer, columnCount, &mean, exampleCount);
		
		_mu[column] = mean;
		
		// `sigma2` is actually a (diagonal) covariance matrix.
		_sigma2[column * columnCount + column] = LNK_pow(LNK_vsd(columnPointer, exampleCount, columnCount, workgroup, mean, NO), 2);
	}
	
	free(workgroup);
}

- (void)dealloc {
	if (_mu)
		free(_mu);
	
	if (_sigma2)
		free(_sigma2);
	
	[super dealloc];
}

@end
