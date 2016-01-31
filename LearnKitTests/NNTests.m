//
//  NNTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>

#import "LNKConfusionMatrix.h"
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
	
	NSArray<LNKNeuralNetLayer *> *hiddenLayers = @[ [[[LNKNeuralNetSigmoidLayer alloc] initWithUnitCount:25] autorelease] ];
	LNKNeuralNetLayer *outputLayer = [[LNKNeuralNetSigmoidLayer alloc] initWithClasses:[LNKClasses withRange:NSMakeRange(1, 10)]];
	
	LNKNeuralNetClassifier *classifier = [[LNKNeuralNetClassifier alloc] initWithMatrix:matrix
																	 implementationType:LNKImplementationTypeAccelerate
																  optimizationAlgorithm:algorithm
																		   hiddenLayers:hiddenLayers
																			outputLayer:outputLayer];
	
	[matrix release];
	[algorithm release];
	[outputLayer release];
	
	NSData *thetaVectorData1 = LNKLoadBinaryMatrixFromFileAtURL([NSURL fileURLWithPath:theta1Path], 401 * 25 * sizeof(double));
	NSData *thetaVectorData2 = LNKLoadBinaryMatrixFromFileAtURL([NSURL fileURLWithPath:theta2Path],  26 * 10 * sizeof(double));
	
	const LNKFloat *thetaVector1 = (const LNKFloat *)thetaVectorData1.bytes;
	const LNKFloat *thetaVector2 = (const LNKFloat *)thetaVectorData2.bytes;
	[classifier _setThetaVector:thetaVector1 transpose:YES forLayerAtIndex:0 rows:25 columns:401];
	[classifier _setThetaVector:thetaVector2 transpose:YES forLayerAtIndex:1 rows:10 columns:26];
	
	return [classifier autorelease];
}

- (void)test1FeedForwardPrediction {
	LNKClassifier *classifier = [self _preLearnedClassifierWithRegularization:NO];
	[self measureBlock:^{
		XCTAssertGreaterThanOrEqual([classifier computeClassificationAccuracyOnTrainingMatrix], 0.97, @"Unexpectedly low classification rate");
	}];
}

- (void)test2CostFunction {
	LNKClassifier *classifier = [self _preLearnedClassifierWithRegularization:NO];
	[self measureBlock:^{
		XCTAssertEqualWithAccuracy(0.287629, [classifier _evaluateCostFunction], DACCURACY, @"Incorrect cost");
	}];
}

- (void)test3CostFunctionRegularized {
	LNKClassifier *classifier = [self _preLearnedClassifierWithRegularization:YES];
	[self measureBlock:^{
		XCTAssertEqualWithAccuracy(0.383770, [classifier _evaluateCostFunction], DACCURACY, @"Incorrect cost");
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
	
	NSArray<LNKNeuralNetLayer *> *hiddenLayers = @[ [[[LNKNeuralNetSigmoidLayer alloc] initWithUnitCount:25] autorelease] ];
	LNKNeuralNetLayer *outputLayer = [[LNKNeuralNetSigmoidLayer alloc] initWithClasses:[LNKClasses withRange:NSMakeRange(1, 10)]];
	
	LNKNeuralNetClassifier *classifier = [[LNKNeuralNetClassifier alloc] initWithMatrix:matrix
																	 implementationType:LNKImplementationTypeAccelerate
																  optimizationAlgorithm:algorithm
																		   hiddenLayers:hiddenLayers
																			outputLayer:outputLayer];
	
	[matrix release];
	[algorithm release];
	[outputLayer release];
	
	[classifier train];
	
	XCTAssertGreaterThanOrEqual([classifier computeClassificationAccuracyOnTrainingMatrix], 0.95, @"Poor accuracy");
	[classifier release];
}

- (void)test5ConfusionMatrix {
	LNKClassifier *const classifier = [self _preLearnedClassifierWithRegularization:NO];
	LNKMatrix *const matrix = classifier.matrix;
	LNKConfusionMatrix *const confusionMatrix = [classifier computeConfusionMatrixOnMatrix:matrix];
	LNKClass *const eight = [LNKClass classWithUnsignedInteger:8];
	const LNKSize examples = matrix.exampleCount / 10;
	XCTAssertGreaterThanOrEqual([confusionMatrix frequencyForTrueClass:eight predictedClass:eight], 0.8 * examples);
}

@end
