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
	NSString *path = [[NSBundle bundleForClass:self.class] pathForResource:@"Cancer" ofType:@"csv"];
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:[NSURL fileURLWithPath:path] addingOnesColumn:YES];
	
	LNKMatrix *trainingMatrix = [matrix shuffledSubmatrixWithExampleCount:483];
	LNKMatrix *testMatrix = [matrix shuffledSubmatrixWithExampleCount:100];
	
	const LNKSize epochs = 50;
	LNKOptimizationAlgorithmStochasticGradientDescent *sgd = [LNKOptimizationAlgorithmStochasticGradientDescent algorithmWithAlpha:[LNKDecayingAlpha withFunction:^LNKFloat(LNKSize iteration) {
		return 1.0/(0.01 * iteration + 50);
	}] iterationCount:epochs];
	sgd.lambda = 0.01;
	sgd.stepCount = 100;
	
	LNKSVMClassifier *classifier = [[LNKSVMClassifier alloc] initWithMatrix:trainingMatrix
														 implementationType:LNKImplementationTypeAccelerate
													  optimizationAlgorithm:sgd
																	classes:[LNKClasses withCount:2]];
	[classifier train];
	
	const LNKFloat accuracy = [classifier computeClassificationAccuracyOnMatrix:testMatrix];
	XCTAssertGreaterThanOrEqual(accuracy, 0.9, "Poor accuracy");
	
	[classifier release];
	[matrix release];
}

@end
