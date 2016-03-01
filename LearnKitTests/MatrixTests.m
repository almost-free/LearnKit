//
//  MatrixTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import <XCTest/XCTest.h>

#import "LNKMatrix.h"
#import "LNKMatrixPrivate.h"

@interface MatrixTests : XCTestCase

@end


@implementation MatrixTests

- (void)testSubmatrices {
	const LNKSize rowCount = 4;
	const LNKSize columnCount = 4;
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithRowCount:4 columnCount:4 addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
#pragma unused(outputVector)
		for (LNKSize example = 0; example < rowCount; example++) {
			for (LNKSize column = 0; column < columnCount; column++) {
				matrix[example * columnCount + column] = example * column;
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
	NSString *const path = [[NSBundle bundleForClass:self.class] pathForResource:@"Pima" ofType:@"csv"];
	LNKMatrix *const matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:[NSURL fileURLWithPath:path] addingOnesColumn:NO];

	const LNKSize rowCount = matrix.rowCount;
	XCTAssertEqual(rowCount, (LNKSize)768);

	LNKSize *const indices = [matrix _shuffleIndices];

	for (LNKSize example = 0; example < rowCount; example++) {
		BOOL okay = NO;
		for (LNKSize local = 0; local < rowCount; local++) {
			if (indices[local] == example) {
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

@end
