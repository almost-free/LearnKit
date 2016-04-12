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
#import "LNKRegularizationConfiguration.h"
#import "LNKUtilities.h"

@interface CollaborativeFilteringTests : XCTestCase

@end

@implementation CollaborativeFilteringTests

- (void)_runCoFiWithAlgorithm:(LNKOptimizationAlgorithmCG *)algorithm lambda:(LNKFloat)lambda test:(void(^)(LNKFloat, const LNKFloat *))testBlock {
	const LNKSize movieCount = 1682;
	const LNKSize userCount = 943;
	const LNKSize rowCount = 10;
	
	const LNKSize reducedMovieCount = 5;
	const LNKSize reducedUserCount = 4;
	const LNKSize reducedRowCount = 3;
	
	NSBundle *bundle = [NSBundle bundleForClass:[self class]];
	NSURL *urlX = [bundle URLForResource:@"Movies_X" withExtension:@"mat"];
	NSURL *urlY = [bundle URLForResource:@"Movies_Y" withExtension:@"mat"];
	NSURL *urlR = [bundle URLForResource:@"Movies_R" withExtension:@"mat"];
	NSURL *urlTheta = [bundle URLForResource:@"Movies_Theta" withExtension:@"mat"];
	
	LNKMatrix *indicatorMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:urlR matrixValueType:LNKValueTypeDouble
															outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
																	 rowCount:movieCount columnCount:userCount];
	
	LNKMatrix *outputMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:urlY matrixValueType:LNKValueTypeDouble
														 outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
																  rowCount:movieCount columnCount:userCount];
	
	LNKCollaborativeFilteringPredictor *predictor = [[LNKCollaborativeFilteringPredictor alloc] initWithMatrix:[outputMatrix submatrixWithRowCount:reducedMovieCount columnCount:reducedUserCount]
																							   indicatorMatrix:[indicatorMatrix submatrixWithRowCount:reducedMovieCount columnCount:reducedUserCount]
																							implementationType:LNKImplementationTypeAccelerate
																						 optimizationAlgorithm:algorithm
																								  featureCount:reducedRowCount];
	if (lambda > 0) {
		predictor.regularizationConfiguration = [LNKRegularizationConfiguration withLambda:lambda];
	}
	[indicatorMatrix release];
	[outputMatrix release];
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:urlX matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
															rowCount:movieCount columnCount:rowCount];
	[predictor loadDataMatrix:[matrix submatrixWithRowCount:reducedMovieCount columnCount:reducedRowCount]];
	[matrix release];
	
	LNKMatrix *thetaMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:urlTheta matrixValueType:LNKValueTypeDouble
														outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
																 rowCount:userCount columnCount:rowCount];
	[predictor loadThetaMatrix:[thetaMatrix submatrixWithRowCount:reducedUserCount columnCount:reducedRowCount]];
	[thetaMatrix release];
	
	const LNKFloat *gradient = [predictor _computeGradient];
	testBlock([predictor _evaluateCostFunction], gradient);
	free((void *)gradient);
	
	[predictor release];
}

- (void)testCostFunction {
	LNKOptimizationAlgorithmCG *algorithm = [[LNKOptimizationAlgorithmCG alloc] init];
	
	[self _runCoFiWithAlgorithm:algorithm lambda:0 test:^(LNKFloat cost, const LNKFloat *gradient) {
		XCTAssertEqualWithAccuracy(cost, 22.22, 0.1);
		XCTAssertEqualWithAccuracy(gradient[0], -2.52899, 0.01);
	}];
	
	[self _runCoFiWithAlgorithm:algorithm lambda:1.5 test:^(LNKFloat cost, const LNKFloat *gradient) {
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
	NSURL *urlY = [bundle URLForResource:@"Movies_Y" withExtension:@"mat"];
	NSURL *urlR = [bundle URLForResource:@"Movies_R" withExtension:@"mat"];
	
	LNKMatrix *indicatorMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:urlR matrixValueType:LNKValueTypeDouble
															outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
																	 rowCount:movieCount columnCount:userCount];
	
	LNKMatrix *outputMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:urlY matrixValueType:LNKValueTypeDouble
														 outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
																  rowCount:movieCount columnCount:userCount];
	
	LNKOptimizationAlgorithmCG *algorithm = [[LNKOptimizationAlgorithmCG alloc] init];
	algorithm.iterationCount = 100;
	
	LNKCollaborativeFilteringPredictor *predictor = [[LNKCollaborativeFilteringPredictor alloc] initWithMatrix:outputMatrix
																							   indicatorMatrix:indicatorMatrix
																							implementationType:LNKImplementationTypeAccelerate
																						 optimizationAlgorithm:algorithm
																								  featureCount:featureCount];
	predictor.regularizationConfiguration = [LNKRegularizationConfiguration withLambda:10];
	[indicatorMatrix release];
	[outputMatrix release];
	[algorithm release];
	
	[predictor train];
	
	NSURL *movieListURL = [bundle URLForResource:@"MovieIDs" withExtension:@"txt"];
	NSString *movieList = [[NSString alloc] initWithContentsOfURL:movieListURL encoding:NSASCIIStringEncoding error:nil];
	NSArray<NSString *> *movies = [movieList componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
	[movieList release];
	
	[[predictor findTopK:10 predictionsForUser:1] enumerateIndexesUsingBlock:^(NSUInteger index, BOOL *stop) {
#pragma unused(stop)
		NSLog(@"%@", movies[index]);
	}];
	
	[predictor release];
}

@end
