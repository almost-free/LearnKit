//
//  NNMNISTTests.m
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

@interface NNMNISTTests : XCTestCase

@end

@implementation NNMNISTTests

#define DACCURACY 0.01

#define EXAMPLES 60000
#define FEATURES (28 * 28)
#define FEATURES_PLUS_ONE (FEATURES+1)

- (void)_copyMNISTDataWithBasename:(NSString *)basename toMatrixBuffer:(LNKFloat *)buffer outputVector:(LNKFloat *)outputVector {
	NSBundle *bundle = [NSBundle bundleForClass:[self class]];
	NSURL *matrixURL = [bundle URLForResource:[basename stringByAppendingString:@"-images"] withExtension:@"idx3-ubyte"];
	NSURL *labelsURL = [bundle URLForResource:[basename stringByAppendingString:@"-labels"] withExtension:@"idx1-ubyte"];
	
	NSData *matrixData = [NSData dataWithContentsOfURL:matrixURL];
	NSData *labelsData = [NSData dataWithContentsOfURL:labelsURL];
	
	NSAssert(matrixData, @"Cannot load matrix data");
	NSAssert(labelsData, @"Cannot load labels data");
	
	// Matrix data files are padded with 16 bytes.
	uint8 *matrixBytes = (uint8 *)[matrixData bytes] + 16;
	
	// Label data files are padded with 8 bytes.
	uint8 *labelBytes = (uint8 *)[labelsData bytes] + 8;
	
	for (LNKSize i = 0; i < EXAMPLES; i++) {
		for (LNKSize n = 0; n < FEATURES; n++) {
			buffer[i * FEATURES_PLUS_ONE + n + 1 /* offset bias unit */] = (CGFloat)matrixBytes[i * FEATURES + n] / 255.0f;
		}
		
		outputVector[i] = (LNKFloat)labelBytes[i];
	}
}

- (void)test1Training {
	LNKMatrix *trainingMatrix = [[LNKMatrix alloc] initWithExampleCount:EXAMPLES
															columnCount:FEATURES
													   addingOnesColumn:YES
														 prepareBuffers:^BOOL(LNKFloat *matrix, LNKFloat *outputVector) {
															 [self _copyMNISTDataWithBasename:@"train" toMatrixBuffer:matrix outputVector:outputVector];
															 return YES;
														 }];
	
	LNKOptimizationAlgorithmStochasticGradientDescent *algorithm = [LNKOptimizationAlgorithmStochasticGradientDescent algorithmWithAlpha:[LNKFixedAlpha withValue:0.3]
																														  iterationCount:5];
	algorithm.stepCount = 50;
	
	NSArray *hiddenLayers = @[ [[[LNKNeuralNetSigmoidLayer alloc] initWithUnitCount:400] autorelease] ];
	LNKNeuralNetLayer *outputLayer = [[LNKNeuralNetSigmoidLayer alloc] initWithClasses:[LNKClasses withRange:NSMakeRange(0, 10)]];
	
	LNKNeuralNetClassifier *classifier = [[LNKNeuralNetClassifier alloc] initWithMatrix:trainingMatrix
																	 implementationType:LNKImplementationTypeAccelerate
																  optimizationAlgorithm:algorithm
																		   hiddenLayers:hiddenLayers
																			outputLayer:outputLayer];
	
	[trainingMatrix release];
	[algorithm release];
	[outputLayer release];
	
	[classifier train];
	
	XCTAssertGreaterThanOrEqual([classifier computeClassificationAccuracyOnTrainingMatrix], 0.90, @"Poor accuracy");
	[classifier release];
}

@end
