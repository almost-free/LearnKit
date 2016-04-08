//
//  SVMTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import <XCTest/XCTest.h>

#import "LNKCSVColumnRule.h"
#import "LNKMatrix.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKSVMClassifier.h"

@interface SVMTests : XCTestCase
@end

@implementation SVMTests

- (void)test1 {
	NSURL *const url = [[NSBundle bundleForClass:self.class] URLForResource:@"Cancer" withExtension:@"csv"];
	LNKMatrix *const matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:url];
	XCTAssertNotNil(matrix);

	LNKMatrix *const subsetMatrix = [matrix submatrixWithRowRange:NSMakeRange(0, 583)];
	XCTAssertEqual(subsetMatrix.rowCount, (LNKSize)583);
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

- (void)testIncome {
	NSURL *const url = [[NSBundle bundleForClass:self.class] URLForResource:@"income" withExtension:@"csv"];
	LNKMatrix *const unnormalizedMatrix = [[LNKMatrix alloc] initWithCSVFileAtURL:url delimiter:',' ignoringHeader:NO columnPreprocessingRules:@{
		@1: [LNKCSVColumnRule deleteRule],
		@3: [LNKCSVColumnRule deleteRule],
		@5: [LNKCSVColumnRule deleteRule],
		@6: [LNKCSVColumnRule deleteRule],
		@7: [LNKCSVColumnRule deleteRule],
		@8: [LNKCSVColumnRule deleteRule],
		@9: [LNKCSVColumnRule deleteRule],
		@13: [LNKCSVColumnRule deleteRule],
		@14: [LNKCSVColumnRule conversionRuleWithBlock:^LNKFloat(NSString *string) {
			if ([string isEqualToString:@" >50K"]) {
				return 1;
			} else {
				return -1;
			}
		}]
	}];
	XCTAssertNotNil(unnormalizedMatrix);

	LNKMatrix *const matrix = unnormalizedMatrix.normalizedMatrix;
	[unnormalizedMatrix release];

	LNKMatrix *trainingMatrix = nil;
	LNKMatrix *testMatrix = nil;
	[matrix splitIntoTrainingMatrix:&trainingMatrix testMatrix:&testMatrix trainingBias:0.8];

	XCTAssertNotNil(trainingMatrix);
	XCTAssertNotNil(testMatrix);

	const LNKSize epochs = 50;
	LNKOptimizationAlgorithmStochasticGradientDescent *sgd = [LNKOptimizationAlgorithmStochasticGradientDescent algorithmWithAlpha:[LNKDecayingAlpha withFunction:^LNKFloat(LNKSize iteration) {
		return 1.0/(0.01 * iteration + 50);
	}] iterationCount:epochs];
	sgd.lambda = 0.001;
	sgd.stepCount = 300;

	LNKSVMClassifier *const classifier = [[LNKSVMClassifier alloc] initWithMatrix:trainingMatrix
															   implementationType:LNKImplementationTypeAccelerate
															optimizationAlgorithm:sgd
																		  classes:[LNKClasses withCount:2]];
	[classifier train];

	const LNKFloat accuracy = [classifier computeClassificationAccuracyOnMatrix:testMatrix];
	NSLog(@"%s: Accuracy: %g", __PRETTY_FUNCTION__, accuracy);
	XCTAssertGreaterThanOrEqual(accuracy, 0.8, "Poor accuracy");

	[classifier release];
}

@end
