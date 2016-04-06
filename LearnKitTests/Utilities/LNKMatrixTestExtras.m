//
//  LNKMatrixTestExtras.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKMatrixTestExtras.h"

#import "LNKAccelerate.h"

@implementation LNKMatrix (TestExtras)

- (LNKMatrix *)polynomialMatrixOfDegree:(LNKSize)maxDegree {
	NSParameterAssert(maxDegree > 0);
	NSParameterAssert(self.columnCount == 1);
	
	const LNKFloat *const currentBuffer = self.matrixBuffer;
	const LNKSize rowCount = self.rowCount;
	const LNKSize columnCount = maxDegree;
	
	return [[[LNKMatrix alloc] initWithRowCount:rowCount columnCount:columnCount prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
		LNKFloatCopy(outputVector, self.outputVector, rowCount);
		
		for (LNKSize row = 0; row < rowCount; row++) {
			const LNKFloat currentValue = currentBuffer[row];
			
			for (LNKSize degree = 1; degree <= maxDegree; degree++) {
				matrix[row * columnCount + (degree - 1)] = LNK_pow(currentValue, degree);
			}
		}
		
		return YES;
	}] autorelease];
}

static inline LNKSize _columnsInPairwisePolynomialMatrixOfDegree(LNKSize maxDegree) {
	LNKSize columns = 0;
	
	for (LNKSize degree = 1; degree <= maxDegree; degree++) {
		for (LNKSize stepDegree = 0; stepDegree <= degree; stepDegree++) {
			columns++;
		}
	}
	
	return columns;
}

- (LNKMatrix *)pairwisePolynomialMatrixOfDegree:(LNKSize)maxDegree {
	NSParameterAssert(maxDegree);
	
	const LNKFloat *currentBuffer = self.matrixBuffer;
	const LNKSize currentColumnCount = self.columnCount;
	const LNKSize rowCount = self.rowCount;
	const LNKSize columnCount = _columnsInPairwisePolynomialMatrixOfDegree(maxDegree);
	
	return [[[LNKMatrix alloc] initWithRowCount:rowCount columnCount:columnCount prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
		LNKFloatCopy(outputVector, self.outputVector, rowCount);

		LNKSize lastIndex = 0;
		
		for (LNKSize degree = 1; degree <= maxDegree; degree++) {
			for (LNKSize stepDegree = 0; stepDegree <= degree; stepDegree++) {
				for (LNKSize row = 0; row < rowCount; row++) {
					const LNKFloat currentValue = currentBuffer[row * currentColumnCount];
					const LNKFloat currentNextValue = currentBuffer[row * currentColumnCount + 1];
					matrix[row * columnCount + lastIndex] = LNK_pow(currentValue, degree - stepDegree) * LNK_pow(currentNextValue, stepDegree);
				}
				
				lastIndex++;
			}
		}
		
		return YES;
	}] autorelease];
}

@end
