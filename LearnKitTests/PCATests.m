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
															   columnCount:2];

	LNKPCAInformation *const pca = [matrix analyzePrincipalComponents];
	XCTAssertEqualWithAccuracy([pca.rotatedMatrix rowAtIndex:0][0],  1.481274, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotatedMatrix rowAtIndex:1][0], -0.912912, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotatedMatrix rowAtIndex:2][0],  1.212087, DACCURACY);

	[matrix release];
}

- (nullable LNKMatrix *)irisMatrix {
	NSURL *const url = [[NSBundle bundleForClass:self.class] URLForResource:@"Iris" withExtension:@"dat"];
	LNKMatrix *const matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:url delimiter:',' ignoringHeader:NO columnPreprocessingRules:@{
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
	return [matrix autorelease];
}

- (void)testIris {
	LNKMatrix *const matrix = self.irisMatrix;
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

	XCTAssertEqual(pca.eigenvalues.length, (LNKSize)4);
	XCTAssertEqualWithAccuracy(pca.eigenvalues.data[0],  2.8914123, DACCURACY);
	XCTAssertEqualWithAccuracy(pca.eigenvalues.data[1],  0.9212208, DACCURACY);
	XCTAssertEqualWithAccuracy(pca.eigenvalues.data[2],  0.1473533, DACCURACY);
	XCTAssertEqualWithAccuracy(pca.eigenvalues.data[3],  0.0206077, DACCURACY);

	// The first vector has flipped signs, but this doesn't matter. It's a perfectly-valid eigenvector.
	XCTAssertEqualWithAccuracy(-[pca.rotationMatrix rowAtIndex:0][0],  0.5223716, DACCURACY);
	XCTAssertEqualWithAccuracy(-[pca.rotationMatrix rowAtIndex:1][0], -0.2633549, DACCURACY);
	XCTAssertEqualWithAccuracy(-[pca.rotationMatrix rowAtIndex:2][0],  0.5812540, DACCURACY);
	XCTAssertEqualWithAccuracy(-[pca.rotationMatrix rowAtIndex:3][0],  0.5656110, DACCURACY);

	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:0][1], -0.37231836, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:1][1], -0.92555649, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:2][1], -0.02109478, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:3][1], -0.06541577, DACCURACY);

	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:0][2],  0.7210168, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:1][2], -0.2420329, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:2][2], -0.1408923, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:3][2], -0.6338014, DACCURACY);

	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:0][3],  0.2619956, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:1][3], -0.1241348, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:2][3], -0.8011543, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:3][3],  0.5235463, DACCURACY);

	XCTAssertEqualWithAccuracy(-[pca.rotatedMatrix rowAtIndex:0][0], -2.256980633, DACCURACY);
	XCTAssertEqualWithAccuracy(-[pca.rotatedMatrix rowAtIndex:1][0], -2.079459119, DACCURACY);

	LNKMatrix *const newMatrix = [matrix matrixProjectedToDimension:2 withPCAInformation:pca];
	XCTAssertEqualWithAccuracy([newMatrix rowAtIndex:0][0],  5.022448, DACCURACY);
}

- (void)testIrisApproximate {
	LNKMatrix *const matrix = self.irisMatrix;
	XCTAssertNotNil(matrix);

	LNKPCAInformation *const pca = [matrix analyzeApproximatePrincipalComponents:2];
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

	XCTAssertEqual(pca.eigenvalues.length, (LNKSize)4);
	XCTAssertEqualWithAccuracy(pca.eigenvalues.data[0],  0, DACCURACY);
	XCTAssertEqualWithAccuracy(pca.eigenvalues.data[1],  0, DACCURACY);
	XCTAssertEqualWithAccuracy(pca.eigenvalues.data[2],  0, DACCURACY);
	XCTAssertEqualWithAccuracy(pca.eigenvalues.data[3],  0, DACCURACY);

	XCTAssertEqualWithAccuracy(-[pca.rotationMatrix rowAtIndex:0][0],  0.5223743, DACCURACY);
	XCTAssertEqualWithAccuracy(-[pca.rotationMatrix rowAtIndex:1][0], -0.2633483, DACCURACY);
	XCTAssertEqualWithAccuracy(-[pca.rotationMatrix rowAtIndex:2][0],  0.5812542, DACCURACY);
	XCTAssertEqualWithAccuracy(-[pca.rotationMatrix rowAtIndex:3][0],  0.5656115, DACCURACY);

	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:0][1],  0.37231521, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:1][1],  0.92555819, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:2][1],  0.02109048, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:3][1],  0.06541118, DACCURACY);

	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:0][2],  0, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:1][2],  0, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:2][2],  0, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:3][2],  0, DACCURACY);

	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:0][3],  0, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:1][3],  0, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:2][3],  0, DACCURACY);
	XCTAssertEqualWithAccuracy([pca.rotationMatrix rowAtIndex:3][3],  0, DACCURACY);

	XCTAssertEqualWithAccuracy(-[pca.rotatedMatrix rowAtIndex:0][0], -2.256980633, DACCURACY);
	XCTAssertEqualWithAccuracy(-[pca.rotatedMatrix rowAtIndex:1][0], -2.079459119, DACCURACY);

	LNKMatrix *const newMatrix = [matrix matrixProjectedToDimension:2 withPCAInformation:pca];
	XCTAssertEqualWithAccuracy([newMatrix rowAtIndex:0][0],  5.022448, DACCURACY);
}

@end
