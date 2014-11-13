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

- (void)_runCoFiWithAlgorithm:(LNKOptimizationAlgorithmCG *)algorithm test:(void(^)(LNKFloat, const LNKFloat *))testBlock {
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
	
	LNKMatrix *indicatorMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:pathR] matrixValueType:LNKValueTypeDouble
															outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
																 exampleCount:movieCount columnCount:userCount
															 addingOnesColumn:NO];
	
	LNKMatrix *outputMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:pathY] matrixValueType:LNKValueTypeDouble
														 outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
															  exampleCount:movieCount columnCount:userCount
														  addingOnesColumn:NO];
	
	LNKCollaborativeFilteringPredictor *predictor = [[LNKCollaborativeFilteringPredictor alloc] initWithMatrix:[outputMatrix submatrixWithExampleCount:reducedMovieCount columnCount:reducedUserCount]
																							   indicatorMatrix:[indicatorMatrix submatrixWithExampleCount:reducedMovieCount columnCount:reducedUserCount]
																							implementationType:LNKImplementationTypeAccelerate
																						 optimizationAlgorithm:algorithm
																								  featureCount:reducedExampleCount];
	[indicatorMatrix release];
	[outputMatrix release];
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:pathX] matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
														exampleCount:movieCount columnCount:exampleCount
													addingOnesColumn:NO];
	[predictor loadDataMatrix:[matrix submatrixWithExampleCount:reducedMovieCount columnCount:reducedExampleCount]];
	[matrix release];
	
	LNKMatrix *thetaMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:pathTheta] matrixValueType:LNKValueTypeDouble
														outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
															 exampleCount:userCount columnCount:exampleCount
														 addingOnesColumn:NO];
	[predictor loadThetaMatrix:[thetaMatrix submatrixWithExampleCount:reducedUserCount columnCount:reducedExampleCount]];
	[thetaMatrix release];
	
	const LNKFloat *gradient = [predictor _computeGradient];
	testBlock([predictor _evaluateCostFunction], gradient);
	free((void *)gradient);
	
	[predictor release];
}

- (void)testCostFunction {
	LNKOptimizationAlgorithmCG *algorithm = [[LNKOptimizationAlgorithmCG alloc] init];
	
	[self _runCoFiWithAlgorithm:algorithm test:^(LNKFloat cost, const LNKFloat *gradient) {
		XCTAssertEqualWithAccuracy(cost, 22.22, 0.1);
		XCTAssertEqualWithAccuracy(gradient[0], -2.52899, 0.01);
	}];
	
	algorithm.regularizationEnabled = YES;
	algorithm.lambda = 1.5;
	
	[self _runCoFiWithAlgorithm:algorithm test:^(LNKFloat cost, const LNKFloat *gradient) {
		XCTAssertEqualWithAccuracy(cost, 31.44, 0.1);
		XCTAssertEqualWithAccuracy(gradient[0], -0.95596, 0.01);
	}];
	
	[algorithm release];
}

- (void)testTraining {
	const LNKSize movieCount = 1682;
	const LNKSize userCount = 943;
	const LNKSize featureCount = 10;
	
	NSBundle *bundle = [NSBundle bundleForClass:[self class]];
	NSString *pathY = [bundle pathForResource:@"Movies_Y" ofType:@"mat"];
	NSString *pathR = [bundle pathForResource:@"Movies_R" ofType:@"mat"];
	
	LNKMatrix *indicatorMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:pathR] matrixValueType:LNKValueTypeDouble
															outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
																 exampleCount:movieCount columnCount:userCount
															 addingOnesColumn:NO];
	
	LNKMatrix *outputMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:pathY] matrixValueType:LNKValueTypeDouble
														 outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
															  exampleCount:movieCount columnCount:userCount
														  addingOnesColumn:NO];
	
	LNKOptimizationAlgorithmCG *algorithm = [[LNKOptimizationAlgorithmCG alloc] init];
	algorithm.regularizationEnabled = YES;
	algorithm.lambda = 10;
	algorithm.iterationCount = 100;
	
	LNKCollaborativeFilteringPredictor *predictor = [[LNKCollaborativeFilteringPredictor alloc] initWithMatrix:outputMatrix
																							   indicatorMatrix:indicatorMatrix
																							implementationType:LNKImplementationTypeAccelerate
																						 optimizationAlgorithm:algorithm
																								  featureCount:featureCount];
	[indicatorMatrix release];
	[outputMatrix release];
	
	[predictor train];
	
	NSURL *movieListURL = [bundle URLForResource:@"MovieIDs" withExtension:@"txt"];
	NSString *movieList = [[NSString alloc] initWithContentsOfURL:movieListURL encoding:NSASCIIStringEncoding error:nil];
	NSArray *movies = [movieList componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
	[movieList release];
	
	[[predictor findTopK:10 predictionsForUser:1] enumerateIndexesUsingBlock:^(NSUInteger index, BOOL *stop) {
#pragma unused(stop)
		NSLog(@"%@", movies[index]);
	}];
	
	[predictor release];
}

@end
