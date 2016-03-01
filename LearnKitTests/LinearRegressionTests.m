//
//  LinearRegressionTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>

#import "LNKAccelerate.h"
#import "LNKAccelerateGradient.h"
#import "LNKLinRegPredictor.h"
#import "LNKLinRegPredictorPrivate.h"
#import "LNKMatrix.h"
#import "LNKMatrixTestExtras.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

@interface LinearRegressionTests : XCTestCase

@end

@implementation LinearRegressionTests

#define DACCURACY 1.0

extern void _LNKComputeBatchGradient(const LNKFloat *matrixBuffer, const LNKFloat *transposeMatrix, const LNKFloat *thetaVector, const LNKFloat *outputVector, LNKFloat *workgroupEC, LNKFloat *workgroupCC, LNKFloat *workgroupCC2, LNKSize rowCount, LNKSize columnCount, BOOL enableRegularization, LNKFloat lambda, LNKHFunction hFunction);

- (LNKLinRegPredictor *)_ex1PredictorGD {
	NSString *path = [[NSBundle bundleForClass:[self class]] pathForResource:@"ex1data1" ofType:@"txt"];
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:[NSURL fileURLWithPath:path] addingOnesColumn:YES];
	id <LNKOptimizationAlgorithm> algorithm = [LNKOptimizationAlgorithmGradientDescent algorithmWithAlpha:[LNKFixedAlpha withValue:0.01]
																						   iterationCount:1500];
	
	LNKLinRegPredictor *predictor = [[LNKLinRegPredictor alloc] initWithMatrix:matrix
															implementationType:LNKImplementationTypeAccelerate
														 optimizationAlgorithm:algorithm];
	[matrix release];
	
	return [predictor autorelease];
}

- (LNKLinRegPredictor *)_ex1PredictorNE {
	NSString *path = [[NSBundle bundleForClass:[self class]] pathForResource:@"ex1data1" ofType:@"txt"];
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:[NSURL fileURLWithPath:path] addingOnesColumn:YES];
	
	LNKOptimizationAlgorithmNormalEquations *algorithm = [[LNKOptimizationAlgorithmNormalEquations alloc] init];
	LNKLinRegPredictor *predictor = [[LNKLinRegPredictor alloc] initWithMatrix:matrix
															implementationType:LNKImplementationTypeAccelerate
														 optimizationAlgorithm:algorithm];
	[algorithm release];
	[matrix release];
	
	return [predictor autorelease];
}

- (void)test1Loading {
	LNKLinRegPredictor *predictor = [self _ex1PredictorGD];
	XCTAssertNotNil(predictor, @"We should have a predictor");
	XCTAssertEqual(predictor.matrix.rowCount, 97UL, @"There should be 97 examples");
	XCTAssertEqual(predictor.matrix.columnCount, 2UL, @"There should be 2 columns: one feature column and the predicted value column");
}

- (void)test2InitialCost {
	XCTAssertEqualWithAccuracy([[self _ex1PredictorGD] _evaluateCostFunction], 32.073, DACCURACY, @"The initial cost should be ~32.07");
}

- (void)test3GradientDescent {
	LNKLinRegPredictor *predictor = [self _ex1PredictorGD];
	[predictor train];
	LNKFloat *thetaVector = [predictor _thetaVector];
	XCTAssertEqualWithAccuracy(thetaVector[0], -3.630291, DACCURACY, @"The Theta vector is incorrect");
	XCTAssertEqualWithAccuracy(thetaVector[1],  1.166362, DACCURACY, @"The Theta vector is incorrect");
}

- (void)test4Prediction {
	LNKLinRegPredictor *predictor = [self _ex1PredictorGD];
	[predictor train];
	LNKFloat input[] = { 1, 3.5 };
	XCTAssertEqualWithAccuracy([[predictor predictValueForFeatureVector:LNKVectorMakeUnsafe(input, 2)] LNKFloatValue] * 10000, 4519.767868, DACCURACY, @"The prediction is incorrect");
}

- (void)test5Normalization {
	NSURL *const url = [[NSBundle bundleForClass:self.class] URLForResource:@"ex1data2" withExtension:@"txt"];
	
	LNKMatrix *const unnormalizedMatrix = [[LNKMatrix alloc] initWithCSVFileAtURL:url addingOnesColumn:YES];
	LNKMatrix *const matrix = unnormalizedMatrix.normalizedMatrix;
	[unnormalizedMatrix release];

	id <LNKOptimizationAlgorithm> algorithm = [LNKOptimizationAlgorithmGradientDescent algorithmWithAlpha:[LNKFixedAlpha withValue:0.01]
																						   iterationCount:400];
	
	LNKLinRegPredictor *predictor = [[LNKLinRegPredictor alloc] initWithMatrix:matrix
															implementationType:LNKImplementationTypeAccelerate
														 optimizationAlgorithm:algorithm];
	[predictor train];
	
	LNKFloat *thetaVector = [predictor _thetaVector];
	XCTAssertEqualWithAccuracy(thetaVector[0], 334302.063993, DACCURACY, @"The Theta vector is incorrect");
	XCTAssertEqualWithAccuracy(thetaVector[1], 100087.116006, DACCURACY, @"The Theta vector is incorrect");
	XCTAssertEqualWithAccuracy(thetaVector[2],   3673.548451, DACCURACY, @"The Theta vector is incorrect");
	
	LNKFloat house[] = { 1, 1650, 3 };
	XCTAssertEqualWithAccuracy([[predictor predictValueForFeatureVector:LNKVectorMakeUnsafe(house, 3)] LNKFloatValue], 289314.618925, DACCURACY, @"The prediction was inaccurate");
	[predictor release];
}

- (void)_testExample1Data2WithAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm {
	NSString *path = [[NSBundle bundleForClass:[self class]] pathForResource:@"ex1data2" ofType:@"txt"];
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:[NSURL fileURLWithPath:path] addingOnesColumn:YES];
	LNKLinRegPredictor *predictor = [[LNKLinRegPredictor alloc] initWithMatrix:matrix
															implementationType:LNKImplementationTypeAccelerate
														 optimizationAlgorithm:algorithm];
	[matrix release];
	[predictor train];
	
	LNKFloat *thetaVector = [predictor _thetaVector];
	XCTAssertEqualWithAccuracy(thetaVector[0], 89597.909540, DACCURACY, @"The Theta vector is incorrect");
	XCTAssertEqualWithAccuracy(thetaVector[1],   139.210674, DACCURACY, @"The Theta vector is incorrect");
	XCTAssertEqualWithAccuracy(thetaVector[2], -8738.019112, DACCURACY, @"The Theta vector is incorrect");
	
	LNKFloat house[] = { 1, 1650, 3 };
	XCTAssertEqualWithAccuracy([[predictor predictValueForFeatureVector:LNKVectorMakeUnsafe(house, 3)] LNKFloatValue], 293081.464335, DACCURACY, @"The prediction was inaccurate");
	[predictor release];
}

- (void)test6NormalEquations {
	LNKOptimizationAlgorithmNormalEquations *algorithm = [[LNKOptimizationAlgorithmNormalEquations alloc] init];
	[self _testExample1Data2WithAlgorithm:algorithm];
	[algorithm release];
}

- (void)test7LBFGS {
	LNKOptimizationAlgorithmLBFGS *algorithm = [[LNKOptimizationAlgorithmLBFGS alloc] init];
	[self _testExample1Data2WithAlgorithm:algorithm];
	[algorithm release];
}

- (void)test8Performance {
	[self measureBlock:^{
		NSString *path = [[NSBundle bundleForClass:[self class]] pathForResource:@"CASP" ofType:@"csv"];
		
		LNKMatrix *matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:[NSURL fileURLWithPath:path] addingOnesColumn:YES];
		LNKOptimizationAlgorithmNormalEquations *algorithm = [[LNKOptimizationAlgorithmNormalEquations alloc] init];
		
		LNKLinRegPredictor *predictor = [[LNKLinRegPredictor alloc] initWithMatrix:matrix
																implementationType:LNKImplementationTypeAccelerate
															 optimizationAlgorithm:algorithm];
		[matrix release];
		[algorithm release];
		
		[predictor train];
		
		LNKFloat x[] = { 1, 12, 6000, 1600, 0.25, 74, 892000, 90, 3000, 80 };
		XCTAssertEqualWithAccuracy([[predictor predictValueForFeatureVector:LNKVectorMakeUnsafe(x, 10)] LNKFloatValue], 37.9766, DACCURACY, @"The prediction was inaccurate");
		[predictor release];
	}];
}

- (void)test9Regularization {
	NSBundle *bundle = [NSBundle bundleForClass:[self class]];
	NSString *xPath = [bundle pathForResource:@"ex5_X" ofType:@"dat"];
	NSString *yPath = [bundle pathForResource:@"ex5_y" ofType:@"dat"];
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:xPath] matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:[NSURL fileURLWithPath:yPath] outputVectorValueType:LNKValueTypeDouble
														rowCount:12 columnCount:1 addingOnesColumn:YES];
	
	LNKOptimizationAlgorithmLBFGS *algorithm = [[LNKOptimizationAlgorithmLBFGS alloc] init];
	algorithm.lambda = 1;
	
	LNKLinRegPredictor *predictor = [[LNKLinRegPredictor alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:algorithm];
	[matrix release];
	[algorithm release];
	
	const LNKFloat thetaVector[2] = { 1, 1 };
	[predictor _setThetaVector:thetaVector];
	
	XCTAssertEqualWithAccuracy([predictor _evaluateCostFunction], 303.9932, DACCURACY, @"The initial cost should be ~303.9932");
	
	const BOOL regularizationEnabled = algorithm.regularizationEnabled;
	const LNKFloat lambda = algorithm.lambda;
	
	const LNKSize rowCount = matrix.rowCount;
	const LNKSize columnCount = matrix.columnCount;
	
	const LNKFloat *matrixBuffer = matrix.matrixBuffer;
	const LNKFloat *outputVector = matrix.outputVector;
	
	LNKFloat *workgroupEC = LNKFloatAlloc(rowCount);
	LNKFloat *workgroupCC = LNKFloatAlloc(columnCount);
	LNKFloat *workgroupCC2 = LNKFloatAlloc(columnCount);
	
	LNKFloat *transposeMatrix = LNKFloatAlloc(rowCount * columnCount);
	LNK_mtrans(matrixBuffer, transposeMatrix, columnCount, rowCount);
	
	_LNKComputeBatchGradient(matrixBuffer, transposeMatrix, thetaVector, outputVector, workgroupEC, workgroupCC, workgroupCC2, rowCount, columnCount, regularizationEnabled, lambda, NULL);
	XCTAssertEqualWithAccuracy(workgroupCC[0], -15.3030, DACCURACY, @"Incorrect gradient");
	XCTAssertEqualWithAccuracy(workgroupCC[1], 598.2507, DACCURACY, @"Incorrect gradient");
	
	free(workgroupEC);
	free(workgroupCC);
	free(workgroupCC2);
	free(transposeMatrix);
	
	[predictor train];
	
	XCTAssertEqualWithAccuracy([predictor _thetaVector][0], 13.08790, DACCURACY, @"Incorrect theta");
	XCTAssertEqualWithAccuracy([predictor _thetaVector][1],  0.36778, DACCURACY, @"Incorrect theta");
	
	[predictor release];
}

- (void)test10LearningCurves {
	NSBundle *bundle = [NSBundle bundleForClass:[self class]];
	NSString *xPath = [bundle pathForResource:@"ex5_X" ofType:@"dat"];
	NSString *yPath = [bundle pathForResource:@"ex5_y" ofType:@"dat"];
	NSString *xcvPath = [bundle pathForResource:@"ex5_Xcv" ofType:@"dat"];
	NSString *ycvPath = [bundle pathForResource:@"ex5_ycv" ofType:@"dat"];
	
	const LNKSize rowCount = 12;
	
	LNKMatrix *trainingSet = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:xPath] matrixValueType:LNKValueTypeDouble
														outputVectorAtURL:[NSURL fileURLWithPath:yPath] outputVectorValueType:LNKValueTypeDouble
															 rowCount:rowCount columnCount:1 addingOnesColumn:YES];
	
	LNKMatrix *cvSet = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:xcvPath] matrixValueType:LNKValueTypeDouble
												  outputVectorAtURL:[NSURL fileURLWithPath:ycvPath] outputVectorValueType:LNKValueTypeDouble
													   rowCount:21 columnCount:1 addingOnesColumn:YES];
	
	LNKFloat trainError[rowCount];
	LNKFloat cvError[rowCount];
	
	LNKOptimizationAlgorithmLBFGS *cvAlgorithm = [[LNKOptimizationAlgorithmLBFGS alloc] init];
	
	for (LNKSize i = 1; i <= rowCount; i++) {
		[trainingSet clipExampleCountTo:i];
		
		LNKOptimizationAlgorithmLBFGS *algorithm = [[LNKOptimizationAlgorithmLBFGS alloc] init];
		algorithm.lambda = 0;
		
		LNKLinRegPredictor *trainPredictor = [[LNKLinRegPredictor alloc] initWithMatrix:trainingSet implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:algorithm];
		[trainPredictor train];
		
		// When predicting, we don't want to regularize.
		algorithm.lambda = 0;
		
		trainError[i-1] = [trainPredictor _evaluateCostFunction];
		
		LNKLinRegPredictor *cvPredictor = [[LNKLinRegPredictor alloc] initWithMatrix:cvSet implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:cvAlgorithm];
		[cvPredictor _setThetaVector:[trainPredictor _thetaVector]];
		cvError[i-1] = [cvPredictor _evaluateCostFunction];
		
		[cvPredictor release];
		[trainPredictor release];
		
		[algorithm release];
	}
	
	[cvAlgorithm release];
	
	XCTAssertEqualWithAccuracy( trainError[0],  0.0000, DACCURACY);
	XCTAssertEqualWithAccuracy( trainError[3],  2.8427, DACCURACY);
	XCTAssertEqualWithAccuracy( trainError[7], 18.1729, DACCURACY);
	XCTAssertEqualWithAccuracy(trainError[11], 22.3739, DACCURACY);
	
	XCTAssertEqualWithAccuracy( cvError[0], 205.1211, DACCURACY);
	XCTAssertEqualWithAccuracy( cvError[3],  48.3689, DACCURACY);
	XCTAssertEqualWithAccuracy( cvError[7],  30.8624, DACCURACY);
	XCTAssertEqualWithAccuracy(cvError[11],  29.4338, DACCURACY);
	
	[trainingSet release];
	[cvSet release];
}

- (void)test11LearningCurvesPolynomial {
	NSBundle *bundle = [NSBundle bundleForClass:[self class]];
	NSString *xPath = [bundle pathForResource:@"ex5_X" ofType:@"dat"];
	NSString *yPath = [bundle pathForResource:@"ex5_y" ofType:@"dat"];
	NSString *xcvPath = [bundle pathForResource:@"ex5_Xcv" ofType:@"dat"];
	NSString *ycvPath = [bundle pathForResource:@"ex5_ycv" ofType:@"dat"];
	
	const LNKSize rowCount = 12;
	
	LNKMatrix *trainingSet_ = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:xPath] matrixValueType:LNKValueTypeDouble
														 outputVectorAtURL:[NSURL fileURLWithPath:yPath] outputVectorValueType:LNKValueTypeDouble
															  rowCount:rowCount columnCount:1 addingOnesColumn:YES];
	LNKMatrix *trainingSet__ = [[trainingSet_ polynomialMatrixOfDegree:8] retain];
	LNKMatrix *trainingSet = trainingSet__.normalizedMatrix;
	
	LNKMatrix *cvSet_ = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:xcvPath] matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:[NSURL fileURLWithPath:ycvPath] outputVectorValueType:LNKValueTypeDouble
														rowCount:21 columnCount:1 addingOnesColumn:YES];
	LNKMatrix *cvSet__ = [[cvSet_ polynomialMatrixOfDegree:8] retain];
	LNKMatrix *cvSet = [cvSet__ normalizedMatrixWithMeanVector:trainingSet.normalizationMeanVector standardDeviationVector:trainingSet.normalizationStandardDeviationVector];
	
	LNKFloat trainError[rowCount];
	LNKFloat cvError[rowCount];
	
	LNKOptimizationAlgorithmLBFGS *cvAlgorithm = [[LNKOptimizationAlgorithmLBFGS alloc] init];
	
	for (LNKSize i = 1; i <= rowCount; i++) {
		[trainingSet clipExampleCountTo:i];
		
		LNKOptimizationAlgorithmLBFGS *algorithm = [[LNKOptimizationAlgorithmLBFGS alloc] init];
		algorithm.lambda = 0;
		
		LNKLinRegPredictor *trainPredictor = [[LNKLinRegPredictor alloc] initWithMatrix:trainingSet implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:algorithm];
		[trainPredictor train];
		
		// When predicting, we don't want to regularize.
		algorithm.lambda = 0;
		
		trainError[i-1] = [trainPredictor _evaluateCostFunction];
		
		LNKLinRegPredictor *cvPredictor = [[LNKLinRegPredictor alloc] initWithMatrix:cvSet implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:cvAlgorithm];
		[cvPredictor _setThetaVector:[trainPredictor _thetaVector]];
		cvError[i-1] = [cvPredictor _evaluateCostFunction];
		
		[cvPredictor release];
		[trainPredictor release];
		
		[algorithm release];
	}
	
	[cvAlgorithm release];
	
	XCTAssertEqualWithAccuracy( trainError[0], 0.0020, DACCURACY);
	XCTAssertEqualWithAccuracy( trainError[3], 0.0000, DACCURACY);
	XCTAssertEqualWithAccuracy( trainError[7], 0.0718, DACCURACY);
	XCTAssertEqualWithAccuracy(trainError[11], 0.0437, DACCURACY);
	
	XCTAssertEqualWithAccuracy( cvError[0], 160.7219, DACCURACY);
	XCTAssertEqualWithAccuracy( cvError[3],  61.9384, DACCURACY);
	XCTAssertEqualWithAccuracy( cvError[7],   7.9725, DACCURACY);
	XCTAssertEqualWithAccuracy(cvError[11],  21,      2);
	
	[trainingSet_ release];
	[trainingSet__ release];
	[cvSet_ release];
	[cvSet__ release];
}

@end
