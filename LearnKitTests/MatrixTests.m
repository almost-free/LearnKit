//
//  MatrixTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import <XCTest/XCTest.h>

#import "LNKMatrix.h"

@interface MatrixTests : XCTestCase

@end


@implementation MatrixTests

- (void)testSubmatrices {
	const LNKSize exampleCount = 4;
	const LNKSize columnCount = 4;
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithExampleCount:4 columnCount:4 addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
#pragma unused(outputVector)
		for (LNKSize example = 0; example < exampleCount; example++) {
			for (LNKSize column = 0; column < columnCount; column++) {
				matrix[example * columnCount + column] = example * column;
			}
		}
		
		return YES;
	}];
	
	LNKMatrix *submatrix = [matrix submatrixWithExampleCount:2 columnCount:3];
	[matrix release];
	
	XCTAssertEqual(submatrix.exampleCount, 2UL);
	XCTAssertEqual(submatrix.columnCount, 3UL);
	
	XCTAssertEqual([submatrix exampleAtIndex:0][0], 0);
	XCTAssertEqual([submatrix exampleAtIndex:0][1], 0);
	XCTAssertEqual([submatrix exampleAtIndex:0][2], 0);
	
	XCTAssertEqual([submatrix exampleAtIndex:1][0], 0);
	XCTAssertEqual([submatrix exampleAtIndex:1][1], 1);
	XCTAssertEqual([submatrix exampleAtIndex:1][2], 2);
}

@end
