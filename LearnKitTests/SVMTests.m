//
//  SVMTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import <XCTest/XCTest.h>

#import "LNKMatrix.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKSVMClassifier.h"

@interface SVMTests : XCTestCase

@end

@implementation SVMTests

- (void)test1 {
	NSString *const path = [[NSBundle bundleForClass:self.class] pathForResource:@"Cancer" ofType:@"csv"];
	LNKMatrix *const matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:[NSURL fileURLWithPath:path] addingOnesColumn:YES];

	LNKMatrix *const subsetMatrix = [matrix submatrixWithExampleRange:NSMakeRange(0, 583)];
	XCTAssertEqual(subsetMatrix.exampleCount, (LNKSize)583);
	XCTAssertEqual(subsetMatrix.columnCount, matrix.columnCount);
	[matrix release];

	LNKMatrix *trainingMatrix = nil;
	LNKMatrix *testMatrix = nil;
	[subsetMatrix splitIntoTrainingMatrix:&trainingMatrix testMatrix:&testMatrix trainingBias:0.83];
	XCTAssertNotNil(trainingMatrix);
	XCTAssertNotNil(testMatrix);

	const LNKSize epochs = 50;
	LNKOptimizationAlgorithmStochasticGradientDescent *sgd = [LNKOptimizationAlgorithmStochasticGradientDescent algorithmWithAlpha:[LNKDecayingAlpha withFunction:^LNKFloat(LNKSize iteration) {
		return 1.0/(0.01 * iteration + 50);
	}] iterationCount:epochs];
	sgd.lambda = 0.01;
	sgd.stepCount = 100;

	LNKSVMClassifier *const classifier = [[LNKSVMClassifier alloc] initWithMatrix:trainingMatrix
															   implementationType:LNKImplementationTypeAccelerate
															optimizationAlgorithm:sgd
																		  classes:[LNKClasses withCount:2]];
	[classifier train];

	const LNKFloat accuracy = [classifier computeClassificationAccuracyOnMatrix:testMatrix];
	XCTAssertGreaterThanOrEqual(accuracy, 0.9, "Poor accuracy");

	[classifier release];
}

@end
