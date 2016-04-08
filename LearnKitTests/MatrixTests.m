//
//  MatrixTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import <XCTest/XCTest.h>

#import "LNKCSVColumnRule.h"
#import "LNKMatrixCSV.h"
#import "LNKMatrixPrivate.h"

@interface MatrixTests : XCTestCase

@end


@implementation MatrixTests

- (void)testSubmatrices {
	const LNKSize rowCount = 4;
	const LNKSize columnCount = 4;
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithRowCount:4 columnCount:4 prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
#pragma unused(outputVector)
		for (LNKSize row = 0; row < rowCount; row++) {
			for (LNKSize column = 0; column < columnCount; column++) {
				matrix[row * columnCount + column] = row * column;
			}
		}
		
		return YES;
	}];
	
	LNKMatrix *submatrix = [matrix submatrixWithRowCount:2 columnCount:3];
	[matrix release];
	
	XCTAssertEqual(submatrix.rowCount, 2UL);
	XCTAssertEqual(submatrix.columnCount, 3UL);
	
	XCTAssertEqual([submatrix rowAtIndex:0][0], 0);
	XCTAssertEqual([submatrix rowAtIndex:0][1], 0);
	XCTAssertEqual([submatrix rowAtIndex:0][2], 0);
	
	XCTAssertEqual([submatrix rowAtIndex:1][0], 0);
	XCTAssertEqual([submatrix rowAtIndex:1][1], 1);
	XCTAssertEqual([submatrix rowAtIndex:1][2], 2);
}

- (void)testShufflingIndices {
	NSURL *const url = [[NSBundle bundleForClass:self.class] URLForResource:@"Pima" withExtension:@"csv"];
	LNKMatrix *const matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:url];

	const LNKSize rowCount = matrix.rowCount;
	XCTAssertEqual(rowCount, (LNKSize)768);

	LNKSize *const indices = [matrix _shuffleIndices];

	for (LNKSize row = 0; row < rowCount; row++) {
		BOOL okay = NO;
		for (LNKSize local = 0; local < rowCount; local++) {
			if (indices[local] == row) {
				okay = YES;
				break;
			}
		}

		if (!okay) {
			XCTFail(@"An index went missing");
		}
	}

	free(indices);
}

- (void)testInversion {
	LNKMatrix *const matrix = [[LNKMatrix alloc] initWithRowCount:2 columnCount:2 prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
#pragma unused(outputVector)
		matrix[0] = 0.5;
		matrix[1] = 2;
		matrix[2] = 3;
		matrix[3] = 0.5;
		return YES;
	}];

	LNKMatrix *const inverse = matrix.invertedMatrix;
	LNKMatrix *const product = [inverse multiplyByMatrix:matrix];
	XCTAssertEqualWithAccuracy(product.matrixBuffer[0],  1, 0.001);
	XCTAssertEqualWithAccuracy(product.matrixBuffer[1],  0, 0.001);
	XCTAssertEqualWithAccuracy(product.matrixBuffer[2],  0, 0.001);
	XCTAssertEqualWithAccuracy(product.matrixBuffer[3],  1, 0.001);

	LNKMatrix *const eye = [[LNKMatrix alloc] initIdentityWithColumnCount:2];
	XCTAssertEqualObjects(product, eye);
	[eye release];
	[matrix release];
}

- (void)testLoadingCSVWithCommas {
	NSURL *const mtcarsURL = [[NSBundle bundleForClass:self.class] URLForResource:@"mtcars_comma" withExtension:@"txt"];

	// Only keep mpg and drat
	LNKMatrix *const matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:mtcarsURL delimiter:',' ignoringHeader:YES columnPreprocessingRules:@{
		@0: [LNKCSVColumnRule deleteRule],
		@1: [LNKCSVColumnRule outputRule],
		@2: [LNKCSVColumnRule deleteRule],
		@3: [LNKCSVColumnRule deleteRule],
		@4: [LNKCSVColumnRule deleteRule],
		@6: [LNKCSVColumnRule deleteRule],
		@7: [LNKCSVColumnRule deleteRule],
		@8: [LNKCSVColumnRule deleteRule],
		@9: [LNKCSVColumnRule deleteRule],
		@10: [LNKCSVColumnRule deleteRule],
		@11: [LNKCSVColumnRule deleteRule]
	}];
	XCTAssertNotNil(matrix);
	[matrix release];
}

@end
