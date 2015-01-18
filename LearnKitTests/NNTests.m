//
//  NNTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>

#import "LNKMatrix.h"
#import "LNKNeuralNetClassifier.h"
#import "LNKNeuralNetClassifierPrivate.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"
#import "LNKUtilities.h"

@interface NNTests : XCTestCase

@end

@implementation NNTests

#define DACCURACY 0.01

- (LNKNeuralNetClassifier *)_preLearnedClassifierWithRegularization:(BOOL)regularize {
	NSBundle *bundle = [NSBundle bundleForClass:[self class]];
	NSString *matrixPath = [bundle pathForResource:@"ex3data1_X" ofType:@"dat"];
	NSString *outputVectorPath = [bundle pathForResource:@"ex3data1_y" ofType:@"dat"];
	NSString *theta1Path = [bundle pathForResource:@"ex3data1_theta1" ofType:@"dat"];
	NSString *theta2Path = [bundle pathForResource:@"ex3data1_theta2" ofType:@"dat"];
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:matrixPath]
													 matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:[NSURL fileURLWithPath:outputVectorPath]
											   outputVectorValueType:LNKValueTypeUInt8
														exampleCount:5000 columnCount:400 addingOnesColumn:YES];
	
	LNKOptimizationAlgorithmCG *algorithm = [[LNKOptimizationAlgorithmCG alloc] init];
	
	if (regularize)
		algorithm.lambda = 1;
	
	const LNKSize hiddenLayerCount = 1;
	LNKNeuralNetLayer hiddenLayers[hiddenLayerCount];
	hiddenLayers[0].unitCount = 25;
	
	LNKNeuralNetClassifier *classifier = [[LNKNeuralNetClassifier alloc] initWithMatrix:matrix
																	 implementationType:LNKImplementationTypeAccelerate
																  optimizationAlgorithm:algorithm
																		   hiddenLayers:hiddenLayers
																	   hiddenLayerCount:hiddenLayerCount
																				classes:[LNKClasses withRange:NSMakeRange(1, 10)]];
	
	[matrix release];
	[algorithm release];
	
	NSData *thetaVectorData1 = LNKLoadBinaryMatrixFromFileAtURL([NSURL fileURLWithPath:theta1Path], 401 * 25 * sizeof(double));
	NSData *thetaVectorData2 = LNKLoadBinaryMatrixFromFileAtURL([NSURL fileURLWithPath:theta2Path],  26 * 10 * sizeof(double));
	
	const LNKFloat *thetaVector1 = (const LNKFloat *)thetaVectorData1.bytes;
	const LNKFloat *thetaVector2 = (const LNKFloat *)thetaVectorData2.bytes;
	[classifier _setThetaVector:thetaVector1 transpose:YES forLayerAtIndex:0 rows:25 columns:401];
	[classifier _setThetaVector:thetaVector2 transpose:YES forLayerAtIndex:1 rows:10 columns:26];
	
	return [classifier autorelease];
}

- (void)test1FeedForwardPrediction {
	[self measureBlock:^{
		XCTAssertGreaterThanOrEqual([[self _preLearnedClassifierWithRegularization:NO] computeClassificationAccuracyOnTrainingMatrix], 0.97, @"Unexpectedly low classification rate");
	}];
}

- (void)test2CostFunction {
	[self measureBlock:^{
		XCTAssertEqualWithAccuracy(0.287629, [[self _preLearnedClassifierWithRegularization:NO] _evaluateCostFunction], DACCURACY, @"Incorrect cost");
	}];
}

- (void)test3CostFunctionRegularized {
	[self measureBlock:^{
		XCTAssertEqualWithAccuracy(0.383770, [[self _preLearnedClassifierWithRegularization:YES] _evaluateCostFunction], DACCURACY, @"Incorrect cost");
	}];
}

- (void)test4Training {
	NSBundle *bundle = [NSBundle bundleForClass:[self class]];
	NSString *matrixPath = [bundle pathForResource:@"ex3data1_X" ofType:@"dat"];
	NSString *outputVectorPath = [bundle pathForResource:@"ex3data1_y" ofType:@"dat"];
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:matrixPath]
													 matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:[NSURL fileURLWithPath:outputVectorPath]
											   outputVectorValueType:LNKValueTypeUInt8
														exampleCount:5000 columnCount:400 addingOnesColumn:YES];
	
	LNKOptimizationAlgorithmCG *algorithm = [[LNKOptimizationAlgorithmCG alloc] init];
	algorithm.iterationCount = 400;
	
	const LNKSize hiddenLayerCount = 1;
	LNKNeuralNetLayer hiddenLayers[hiddenLayerCount];
	hiddenLayers[0].unitCount = 25;
	
	LNKNeuralNetClassifier *classifier = [[LNKNeuralNetClassifier alloc] initWithMatrix:matrix
																	 implementationType:LNKImplementationTypeAccelerate
																  optimizationAlgorithm:algorithm
																		   hiddenLayers:hiddenLayers
																	   hiddenLayerCount:hiddenLayerCount
																				classes:[LNKClasses withRange:NSMakeRange(1, 10)]];
	
	[matrix release];
	[algorithm release];
	
	[classifier train];
	
	XCTAssertGreaterThanOrEqual([classifier computeClassificationAccuracyOnTrainingMatrix], 0.97, @"Poor accuracy");
	[classifier release];
}

@end
