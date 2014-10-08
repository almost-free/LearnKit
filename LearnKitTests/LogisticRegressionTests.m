//
//  LogisticRegressionTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>

#import "LNKLogRegClassifier.h"
#import "LNKLogRegClassifierPrivate.h"
#import "LNKMatrixTestExtras.h"
#import "LNKOneVsAllLogRegClassifier.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

@interface LogisticRegressionTests : XCTestCase

@end

@implementation LogisticRegressionTests

#define DACCURACY 0.01

- (void)test1 {
	NSString *path = [[NSBundle bundleForClass:[self class]] pathForResource:@"ex2data1" ofType:@"csv"];
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:[NSURL fileURLWithPath:path] addingOnesColumn:YES];
	LNKOptimizationAlgorithmLBFGS *algorithm = [[LNKOptimizationAlgorithmLBFGS alloc] init];
	LNKLogRegClassifier *classifier = [[LNKLogRegClassifier alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:algorithm];
	XCTAssertEqualWithAccuracy([classifier _evaluateCostFunction], 0.693147, DACCURACY, @"Incorrect cost");
	
	[matrix release];
	[algorithm release];
	
	[classifier train];
	
	LNKFloat *theta = [classifier _thetaVector];
	XCTAssertEqualWithAccuracy(theta[0], -25.171272, DACCURACY, @"Incorrect theta vector");
	XCTAssertEqualWithAccuracy(theta[1], 0.206233, DACCURACY, @"Incorrect theta vector");
	XCTAssertEqualWithAccuracy(theta[2], 0.201470, DACCURACY, @"Incorrect theta vector");
	
	XCTAssertEqualWithAccuracy([classifier _evaluateCostFunction], 0.203498, DACCURACY, @"Incorrect cost");
	
	LNKFloat inputVector[3] = {1,45,85};
	XCTAssertEqualWithAccuracy([[classifier predictValueForFeatureVector:inputVector length:3] LNKFloatValue], 0.776, DACCURACY, @"Incorrect prediction");
	
	XCTAssertEqualWithAccuracy([classifier computeClassificationAccuracy], 0.89, DACCURACY, @"Incorrect classification rate");
	[classifier release];
}

- (void)_testRegularizationWithLambda:(LNKFloat)lambda cost:(LNKFloat)cost {
	NSString *path = [[NSBundle bundleForClass:[self class]] pathForResource:@"ex2data2" ofType:@"txt"];
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:[NSURL fileURLWithPath:path] addingOnesColumn:YES];
	LNKMatrix *polynomialMatrix = [matrix pairwisePolynomialMatrixOfDegree:6];
	XCTAssertEqual(polynomialMatrix.columnCount, 28UL, @"We should have 28 columns");
	
	LNKOptimizationAlgorithmLBFGS *algorithm = [[LNKOptimizationAlgorithmLBFGS alloc] init];
	algorithm.regularizationEnabled = YES;
	algorithm.lambda = lambda;
	
	LNKLogRegClassifier *classifier = [[LNKLogRegClassifier alloc] initWithMatrix:polynomialMatrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:algorithm];
	XCTAssertEqualWithAccuracy([classifier _evaluateCostFunction], 0.693147, DACCURACY, @"Incorrect cost");
	
	[matrix release];
	[algorithm release];
	
	[classifier train];
	
	XCTAssertEqualWithAccuracy([classifier _evaluateCostFunction], cost, DACCURACY, @"Incorrect cost");
	[classifier release];
}

- (void)test2Regularization {
	[self _testRegularizationWithLambda:1 cost:0.52900];
}

- (void)test2Regularization2 {
	[self _testRegularizationWithLambda:100 cost:0.68648];
}

- (void)test3OneVsAll {
	NSString *matrixPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"ex3data1_X" ofType:@"dat"];
	NSString *outputVectorPath = [[NSBundle bundleForClass:[self class]] pathForResource:@"ex3data1_y" ofType:@"dat"];
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:matrixPath]
													 matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:[NSURL fileURLWithPath:outputVectorPath]
											   outputVectorValueType:LNKValueTypeUInt8
														exampleCount:5000 columnCount:400 addingOnesColumn:YES];
	
	XCTAssertEqual(matrix.columnCount, 401ULL, @"The ones column was not added");
	
	LNKOptimizationAlgorithmLBFGS *algorithm = [[LNKOptimizationAlgorithmLBFGS alloc] init];
	algorithm.regularizationEnabled = YES;
	algorithm.lambda = 0.1;
	
	LNKOneVsAllLogRegClassifier *classifier = [[LNKOneVsAllLogRegClassifier alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:algorithm classes:[LNKClasses withRange:NSMakeRange(1, 10)]];
	[classifier train];
	
	[algorithm release];
	[matrix release];
	
	XCTAssertGreaterThanOrEqual([classifier computeClassificationAccuracy], 0.95, @"Poor success rate");
	[classifier release];
}

- (void)test3OneVsAllPerformance {
	[self measureBlock:^{
		[self test3OneVsAll];
	}];
}

@end
