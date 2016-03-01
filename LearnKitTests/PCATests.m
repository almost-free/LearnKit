//
//  PCATests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>

#import "LNKCSVColumnRule.h"
#import "LNKMatrix.h"
#import "LNKMatrixPCA.h"

@interface PCATests : XCTestCase
@end

@implementation PCATests

#define DACCURACY 0.01

- (void)test1 {
	NSURL *const url = [[NSBundle bundleForClass:self.class] URLForResource:@"ex7data1_X" withExtension:@"dat"];
	LNKMatrix *const matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:url matrixValueType:LNKValueTypeDouble
														 outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
															  rowCount:50
															   columnCount:2
														  addingOnesColumn:NO];

	LNKMatrix *const reducedMatrix = [[matrix matrixReducedToDimension:1] retain];
	XCTAssertEqualWithAccuracy(reducedMatrix.matrixBuffer[0],  1.481274, DACCURACY);
	XCTAssertEqualWithAccuracy(reducedMatrix.matrixBuffer[1], -0.912912, DACCURACY);
	XCTAssertEqualWithAccuracy(reducedMatrix.matrixBuffer[2],  1.212087, DACCURACY);

	[reducedMatrix release];
	[matrix release];
}

- (void)testIris {
	NSURL *const url = [[NSBundle bundleForClass:self.class] URLForResource:@"Iris" withExtension:@"dat"];
	LNKMatrix *const matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:url delimiter:',' addingOnesColumn:NO columnPreprocessingRules:@{
		@4: [LNKCSVColumnRule conversionRuleWithBlock:^LNKFloat(NSString *stringValue) {
			if ([stringValue hasSuffix:@"setosa"]) {
				return 0;
			} else if ([stringValue hasSuffix:@"versicolor"]) {
				return 1;
			} else if ([stringValue hasSuffix:@"virginica"]) {
				return 2;
			} else {
				XCTFail();
				return -1;
			}
		}]
	}];

	XCTAssertNotNil(matrix);

	LNKPCAInformation *pca = [matrix analyzePrincipalComponents];
	XCTAssertNotNil(pca);

	XCTAssertEqual(pca.centers.length, (LNKSize)4);
	XCTAssertEqualWithAccuracy(pca.centers.data[0],  5.843333, DACCURACY);
	XCTAssertEqualWithAccuracy(pca.centers.data[1],  3.054000, DACCURACY);
	XCTAssertEqualWithAccuracy(pca.centers.data[2],  3.758667, DACCURACY);
	XCTAssertEqualWithAccuracy(pca.centers.data[3],  1.198667, DACCURACY);

	XCTAssertEqual(pca.scales.length, (LNKSize)4);
	XCTAssertEqualWithAccuracy(pca.scales.data[0],  0.8280661, DACCURACY);
	XCTAssertEqualWithAccuracy(pca.scales.data[1],  0.4335943, DACCURACY);
	XCTAssertEqualWithAccuracy(pca.scales.data[2],  1.7644204, DACCURACY);
	XCTAssertEqualWithAccuracy(pca.scales.data[3],  0.7631607, DACCURACY);

	XCTAssertEqual(pca.standardDeviations.length, (LNKSize)4);
	XCTAssertEqualWithAccuracy(pca.standardDeviations.data[0],  1.7061120, DACCURACY);
	XCTAssertEqualWithAccuracy(pca.standardDeviations.data[1],  0.9598025, DACCURACY);
	XCTAssertEqualWithAccuracy(pca.standardDeviations.data[2],  0.3838662, DACCURACY);
	XCTAssertEqualWithAccuracy(pca.standardDeviations.data[3],  0.1435538, DACCURACY);

	// The first vector has flipped signs, but this doesn't matter. It's a perfectly-valid eigenvector.
	XCTAssertEqualWithAccuracy(-[pca.rotationMatrix rowAtIndex:0][0],  0.5223716, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:0][1], -0.37231836, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:0][2],  0.7210168, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:0][3],  0.2619956, DACCURACY);

	XCTAssertEqualWithAccuracy(-[pca.rotationMatrix rowAtIndex:1][0], -0.2633549, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:1][1], -0.92555649, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:1][2], -0.2420329, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:1][3], -0.1241348, DACCURACY);

	XCTAssertEqualWithAccuracy(-[pca.rotationMatrix rowAtIndex:2][0],  0.5812540, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:2][1], -0.02109478, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:2][2], -0.1408923, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:2][3], -0.8011543, DACCURACY);

	XCTAssertEqualWithAccuracy(-[pca.rotationMatrix rowAtIndex:3][0],  0.5656110, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:3][1], -0.06541577, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:3][2], -0.6338014, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:3][3],  0.5235463, DACCURACY);

	XCTAssertEqualWithAccuracy(-[pca.rotatedMatrix rowAtIndex:0][0], -2.256980633, DACCURACY);
	XCTAssertEqualWithAccuracy(-[pca.rotatedMatrix rowAtIndex:1][0], -2.079459119, DACCURACY);

	LNKMatrix *const newMatrix = [matrix matrixProjectedToDimension:2];
	XCTAssertEqualWithAccuracy([newMatrix rowAtIndex:0][0],  5.022448, DACCURACY);

	[matrix release];
}

@end
