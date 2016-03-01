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
	NSParameterAssert(maxDegree);
	NSParameterAssert(self.columnCount == 2);
	
	const LNKFloat *currentBuffer = self.matrixBuffer;
	const LNKSize currentColumnCount = self.columnCount;
	
	const LNKSize rowCount = self.rowCount;
	const LNKSize columnCountWithoutOnes = maxDegree;
	const LNKSize hasOnesColumn = 1;
	const LNKSize columnCount = columnCountWithoutOnes + hasOnesColumn;
	
	return [[[LNKMatrix alloc] initWithExampleCount:rowCount columnCount:columnCountWithoutOnes addingOnesColumn:YES prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
		LNKFloatCopy(outputVector, self.outputVector, rowCount);
		
		for (LNKSize row = 0; row < rowCount; row++) {
			// Start after the ones column.
			LNKSize lastIndex = hasOnesColumn;
			
			for (LNKSize degree = 1; degree <= maxDegree; degree++) {
				const LNKFloat currentValue = currentBuffer[row * currentColumnCount + hasOnesColumn];
				matrix[row * columnCount + lastIndex++] = LNK_pow(currentValue, degree);
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
	const LNKSize columnCountWithoutOnes = _columnsInPairwisePolynomialMatrixOfDegree(maxDegree);
	const LNKSize hasOnesColumn = 1;
	const LNKSize columnCount = columnCountWithoutOnes + hasOnesColumn;
	
	return [[[LNKMatrix alloc] initWithExampleCount:rowCount columnCount:columnCountWithoutOnes addingOnesColumn:YES prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
		LNKFloatCopy(outputVector, self.outputVector, rowCount);
		
		// Start after the ones column.
		LNKSize lastIndex = hasOnesColumn;
		
		for (LNKSize degree = 1; degree <= maxDegree; degree++) {
			for (LNKSize stepDegree = 0; stepDegree <= degree; stepDegree++) {
				for (LNKSize row = 0; row < rowCount; row++) {
					const LNKFloat currentValue = currentBuffer[row * currentColumnCount + hasOnesColumn];
					const LNKFloat currentNextValue = currentBuffer[row * currentColumnCount + hasOnesColumn + 1];
					matrix[row * columnCount + lastIndex] = LNK_pow(currentValue, degree - stepDegree) * LNK_pow(currentNextValue, stepDegree);
				}
				
				lastIndex++;
			}
		}
		
		return YES;
	}] autorelease];
}

@end
