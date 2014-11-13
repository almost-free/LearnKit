//
//  CollaborativeFilteringTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import <XCTest/XCTest.h>

#import "LNKCollaborativeFilteringPredictorPrivate.h"
#import "LNKMatrix.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"
#import "LNKUtilities.h"

@interface CollaborativeFilteringTests : XCTestCase

@end

@implementation CollaborativeFilteringTests

- (LNKFloat)_costWithAlgorithm:(LNKOptimizationAlgorithmCG *)algorithm {
	const LNKSize movieCount = 1682;
	const LNKSize userCount = 943;
	const LNKSize exampleCount = 10;
	
	const LNKSize reducedMovieCount = 5;
	const LNKSize reducedUserCount = 4;
	const LNKSize reducedExampleCount = 3;
	
	NSBundle *bundle = [NSBundle bundleForClass:[self class]];
	NSString *pathX = [bundle pathForResource:@"Movies_X" ofType:@"mat"];
	NSString *pathY = [bundle pathForResource:@"Movies_Y" ofType:@"mat"];
	NSString *pathR = [bundle pathForResource:@"Movies_R" ofType:@"mat"];
	NSString *pathTheta = [bundle pathForResource:@"Movies_Theta" ofType:@"mat"];
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:pathX] matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
														exampleCount:movieCount columnCount:exampleCount
													addingOnesColumn:NO];
	
	LNKCollaborativeFilteringPredictor *predictor = [[LNKCollaborativeFilteringPredictor alloc] initWithMatrix:[matrix submatrixWithExampleCount:reducedMovieCount columnCount:reducedExampleCount]
																							implementationType:LNKImplementationTypeAccelerate
																						 optimizationAlgorithm:algorithm
																									 userCount:reducedUserCount];
	[matrix release];
	
	LNKMatrix *thetaMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:pathTheta] matrixValueType:LNKValueTypeDouble
														outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
															 exampleCount:userCount columnCount:exampleCount
														 addingOnesColumn:NO];
	
	[predictor _setThetaMatrix:[thetaMatrix submatrixWithExampleCount:reducedUserCount columnCount:reducedExampleCount]];
	[thetaMatrix release];
	
	LNKMatrix *indicatorMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:pathR] matrixValueType:LNKValueTypeDouble
															outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
																 exampleCount:movieCount columnCount:userCount
															 addingOnesColumn:NO];
	
	predictor.indicatorMatrix = [indicatorMatrix submatrixWithExampleCount:reducedMovieCount columnCount:reducedUserCount];
	[indicatorMatrix release];
	
	LNKMatrix *outputMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:pathY] matrixValueType:LNKValueTypeDouble
														 outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
															  exampleCount:movieCount columnCount:userCount
														  addingOnesColumn:NO];

	predictor.outputMatrix = [outputMatrix submatrixWithExampleCount:reducedMovieCount columnCount:reducedUserCount];
	[outputMatrix release];
	
	LNKFloat cost = [predictor _evaluateCostFunction];
	[predictor release];
	
	return cost;
}

- (void)testCostFunction {
	LNKOptimizationAlgorithmCG *algorithm = [[LNKOptimizationAlgorithmCG alloc] init];
	
	LNKFloat cost = [self _costWithAlgorithm:algorithm];
	XCTAssertEqualWithAccuracy(cost, 22.22, 0.1);
	
	algorithm.regularizationEnabled = YES;
	algorithm.lambda = 1.5;
	
	cost = [self _costWithAlgorithm:algorithm];
	XCTAssertEqualWithAccuracy(cost, 31.44, 0.1);
	
	[algorithm release];
}

@end
