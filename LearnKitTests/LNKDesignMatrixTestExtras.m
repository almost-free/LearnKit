//
//  LNKDesignMatrixTestExtras.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKDesignMatrixTestExtras.h"

#import "LNKAccelerate.h"

@implementation LNKDesignMatrix (TestExtras)

- (LNKDesignMatrix *)polynomialMatrixOfDegree:(LNKSize)maxDegree {
	NSParameterAssert(maxDegree);
	NSParameterAssert(self.columnCount == 2);
	
	const LNKFloat *currentBuffer = self.matrixBuffer;
	const LNKSize currentColumnCount = self.columnCount;
	
	const LNKSize exampleCount = self.exampleCount;
	const LNKSize columnCountWithoutOnes = maxDegree;
	const LNKSize hasOnesColumn = 1;
	const LNKSize columnCount = columnCountWithoutOnes + hasOnesColumn;
	
	return [[[LNKDesignMatrix alloc] initWithExampleCount:exampleCount columnCount:columnCountWithoutOnes addingOnesColumn:YES prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
		LNKFloatCopy(outputVector, self.outputVector, exampleCount);
		
		for (LNKSize row = 0; row < exampleCount; row++) {
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

- (LNKDesignMatrix *)pairwisePolynomialMatrixOfDegree:(LNKSize)maxDegree {
	NSParameterAssert(maxDegree);
	
	const LNKFloat *currentBuffer = self.matrixBuffer;
	const LNKSize currentColumnCount = self.columnCount;
	
	const LNKSize exampleCount = self.exampleCount;
	const LNKSize columnCountWithoutOnes = _columnsInPairwisePolynomialMatrixOfDegree(maxDegree);
	const LNKSize hasOnesColumn = 1;
	const LNKSize columnCount = columnCountWithoutOnes + hasOnesColumn;
	
	return [[[LNKDesignMatrix alloc] initWithExampleCount:exampleCount columnCount:columnCountWithoutOnes addingOnesColumn:YES prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
		LNKFloatCopy(outputVector, self.outputVector, exampleCount);
		
		// Start after the ones column.
		LNKSize lastIndex = hasOnesColumn;
		
		for (LNKSize degree = 1; degree <= maxDegree; degree++) {
			for (LNKSize stepDegree = 0; stepDegree <= degree; stepDegree++) {
				for (LNKSize row = 0; row < exampleCount; row++) {
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
