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
	
	NSBundle *bundle = [NSBundle bundleForClass:[self class]];
	NSString *pathX = [bundle pathForResource:@"Movies_X" ofType:@"mat"];
	NSString *pathY = [bundle pathForResource:@"Movies_Y" ofType:@"mat"];
	NSString *pathR = [bundle pathForResource:@"Movies_R" ofType:@"mat"];
	NSString *pathTheta = [bundle pathForResource:@"Movies_Theta" ofType:@"mat"];
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:pathX] matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
														exampleCount:movieCount columnCount:exampleCount
													addingOnesColumn:NO];
	
	LNKCollaborativeFilteringPredictor *predictor = [[LNKCollaborativeFilteringPredictor alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:algorithm userCount:userCount];
	[matrix release];
	
	LNKMatrix *thetaMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:pathTheta] matrixValueType:LNKValueTypeDouble
														outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
															 exampleCount:userCount columnCount:exampleCount
														 addingOnesColumn:NO];
	
	[predictor _setThetaMatrix:thetaMatrix];
	[thetaMatrix release];
	
	LNKMatrix *indicatorMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:pathR] matrixValueType:LNKValueTypeDouble
															outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
																 exampleCount:movieCount columnCount:userCount
															 addingOnesColumn:NO];
	
	predictor.indicatorMatrix = indicatorMatrix;
	[indicatorMatrix release];
	
	LNKMatrix *outputMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:pathY] matrixValueType:LNKValueTypeDouble
														 outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
															  exampleCount:movieCount columnCount:userCount
														  addingOnesColumn:NO];

	predictor.outputMatrix = outputMatrix;
	[outputMatrix release];
	
	LNKFloat cost = [predictor _evaluateCostFunction];
	[predictor release];
	
	return cost;
}

- (void)testCostFunction {
	LNKOptimizationAlgorithmCG *algorithm = [[LNKOptimizationAlgorithmCG alloc] init];
	
	LNKFloat cost = [self _costWithAlgorithm:algorithm];
	XCTAssertEqualWithAccuracy(cost, 27918, 1);
	
	algorithm.regularizationEnabled = YES;
	algorithm.lambda = 1;
	
	cost = [self _costWithAlgorithm:algorithm];
	XCTAssertEqualWithAccuracy(cost, 32520, 1);
	
	[algorithm release];
}

@end
