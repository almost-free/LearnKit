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
#import "LNKCSVColumnRule.h"
#import "LNKLinRegPredictor.h"
#import "LNKLinRegPredictor+Analysis.h"
#import "LNKLinRegPredictorPrivate.h"
#import "LNKMatrixCSV.h"
#import "LNKMatrixTestExtras.h"
#import "LNKOptimizationAlgorithm.h"
#import "LNKPredictorPrivate.h"

@interface LinearRegressionTests : XCTestCase

@end

@implementation LinearRegressionTests

#define DACCURACY 1.0

extern void _LNKComputeBatchGradient(const LNKFloat *matrixBuffer, const LNKFloat *transposeMatrix, const LNKFloat *thetaVector, const LNKFloat *outputVector, LNKFloat *workgroupEC, LNKFloat *workgroupCC, LNKFloat *workgroupCC2, LNKSize rowCount, LNKSize columnCount, BOOL enableRegularization, LNKFloat lambda, LNKHFunction hFunction);

- (LNKLinRegPredictor *)_ex1PredictorGD {
	NSURL *url = [[NSBundle bundleForClass:[self class]] URLForResource:@"ex1data1" withExtension:@"txt"];
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:url];
	id <LNKOptimizationAlgorithm> algorithm = [LNKOptimizationAlgorithmGradientDescent algorithmWithAlpha:[LNKFixedAlpha withValue:0.01]
																						   iterationCount:1500];
	
	LNKLinRegPredictor *predictor = [[LNKLinRegPredictor alloc] initWithMatrix:matrix
															implementationType:LNKImplementationTypeAccelerate
														 optimizationAlgorithm:algorithm];
	[matrix release];
	
	return [predictor autorelease];
}

- (LNKLinRegPredictor *)_ex1PredictorNE {
	NSURL *url = [[NSBundle bundleForClass:[self class]] URLForResource:@"ex1data1" withExtension:@"txt"];
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:url];
	
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
	LNKFloat input[] = { 3.5 };
	XCTAssertEqualWithAccuracy([[predictor predictValueForFeatureVector:LNKVectorMakeUnsafe(input, 1)] LNKFloatValue] * 10000, 4519.767868, DACCURACY, @"The prediction is incorrect");
}

- (void)test5Normalization {
	NSURL *const url = [[NSBundle bundleForClass:self.class] URLForResource:@"ex1data2" withExtension:@"txt"];
	
	LNKMatrix *const unnormalizedMatrix = [[LNKMatrix alloc] initWithCSVFileAtURL:url];
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
	
	LNKFloat house[] = { 1650, 3 };
	XCTAssertEqualWithAccuracy([[predictor predictValueForFeatureVector:LNKVectorMakeUnsafe(house, 2)] LNKFloatValue], 289314.618925, DACCURACY, @"The prediction was inaccurate");
	[predictor release];
}

- (void)_testExample1Data2WithAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm {
	NSURL *url = [[NSBundle bundleForClass:[self class]] URLForResource:@"ex1data2" withExtension:@"txt"];
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:url];
	LNKLinRegPredictor *predictor = [[LNKLinRegPredictor alloc] initWithMatrix:matrix
															implementationType:LNKImplementationTypeAccelerate
														 optimizationAlgorithm:algorithm];
	[matrix release];
	[predictor train];
	
	LNKFloat *thetaVector = [predictor _thetaVector];
	XCTAssertEqualWithAccuracy(thetaVector[0], 89597.909540, DACCURACY, @"The Theta vector is incorrect");
	XCTAssertEqualWithAccuracy(thetaVector[1],   139.210674, DACCURACY, @"The Theta vector is incorrect");
	XCTAssertEqualWithAccuracy(thetaVector[2], -8738.019112, DACCURACY, @"The Theta vector is incorrect");
	
	LNKFloat house[] = { 1650, 3 };
	XCTAssertEqualWithAccuracy([[predictor predictValueForFeatureVector:LNKVectorMakeUnsafe(house, 2)] LNKFloatValue], 293081.464335, DACCURACY, @"The prediction was inaccurate");
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
		NSURL *url = [[NSBundle bundleForClass:[self class]] URLForResource:@"CASP" withExtension:@"csv"];
		
		LNKMatrix *matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:url];
		LNKOptimizationAlgorithmNormalEquations *algorithm = [[LNKOptimizationAlgorithmNormalEquations alloc] init];
		
		LNKLinRegPredictor *predictor = [[LNKLinRegPredictor alloc] initWithMatrix:matrix
																implementationType:LNKImplementationTypeAccelerate
															 optimizationAlgorithm:algorithm];
		[matrix release];
		[algorithm release];
		
		[predictor train];
		
		LNKFloat x[] = { 12, 6000, 1600, 0.25, 74, 892000, 90, 3000, 80 };
		XCTAssertEqualWithAccuracy([[predictor predictValueForFeatureVector:LNKVectorMakeUnsafe(x, 9)] LNKFloatValue], 37.9766, DACCURACY, @"The prediction was inaccurate");
		[predictor release];
	}];
}

- (void)test9Regularization {
	NSBundle *bundle = [NSBundle bundleForClass:[self class]];
	NSURL *xURL = [bundle URLForResource:@"ex5_X" withExtension:@"dat"];
	NSURL *yURL = [bundle URLForResource:@"ex5_y" withExtension:@"dat"];
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:xURL matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:yURL outputVectorValueType:LNKValueTypeDouble
															rowCount:12 columnCount:1];
	
	LNKOptimizationAlgorithmLBFGS *algorithm = [[LNKOptimizationAlgorithmLBFGS alloc] init];
	algorithm.lambda = 1;
	
	LNKLinRegPredictor *predictor = [[LNKLinRegPredictor alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:algorithm];
	[matrix release];
	[algorithm release];

	const LNKFloat thetaVector[2] = { 1, 1 };
	[predictor _setThetaVector:thetaVector];

	LNKMatrix *const workingMatrix = predictor.matrix;
	
	XCTAssertEqualWithAccuracy([predictor _evaluateCostFunction], 303.9932, DACCURACY, @"The initial cost should be ~303.9932");
	
	const BOOL regularizationEnabled = algorithm.regularizationEnabled;
	const LNKFloat lambda = algorithm.lambda;
	
	const LNKSize rowCount = workingMatrix.rowCount;
	const LNKSize columnCount = workingMatrix.columnCount;
	
	const LNKFloat *matrixBuffer = workingMatrix.matrixBuffer;
	const LNKFloat *outputVector = workingMatrix.outputVector;
	
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
	NSURL *xURL = [bundle URLForResource:@"ex5_X" withExtension:@"dat"];
	NSURL *yURL = [bundle URLForResource:@"ex5_y" withExtension:@"dat"];
	NSURL *xcvURL = [bundle URLForResource:@"ex5_Xcv" withExtension:@"dat"];
	NSURL *ycvURL = [bundle URLForResource:@"ex5_ycv" withExtension:@"dat"];
	
	const LNKSize rowCount = 12;
	
	LNKMatrix *trainingSet = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:xURL matrixValueType:LNKValueTypeDouble
														outputVectorAtURL:yURL outputVectorValueType:LNKValueTypeDouble
																 rowCount:rowCount columnCount:1];
	
	LNKMatrix *cvSet = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:xcvURL matrixValueType:LNKValueTypeDouble
												  outputVectorAtURL:ycvURL outputVectorValueType:LNKValueTypeDouble
														   rowCount:21 columnCount:1];
	
	LNKFloat trainError[rowCount];
	LNKFloat cvError[rowCount];
	
	LNKOptimizationAlgorithmLBFGS *cvAlgorithm = [[LNKOptimizationAlgorithmLBFGS alloc] init];
	
	for (LNKSize i = 1; i <= rowCount; i++) {
		[trainingSet clipRowCountTo:i];
		
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
	NSURL *xURL = [bundle URLForResource:@"ex5_X" withExtension:@"dat"];
	NSURL *yURL = [bundle URLForResource:@"ex5_y" withExtension:@"dat"];
	NSURL *xcvURL = [bundle URLForResource:@"ex5_Xcv" withExtension:@"dat"];
	NSURL *ycvURL = [bundle URLForResource:@"ex5_ycv" withExtension:@"dat"];

	const LNKSize rowCount = 12;

	LNKMatrix *trainingSetSingle = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:xURL matrixValueType:LNKValueTypeDouble
															  outputVectorAtURL:yURL outputVectorValueType:LNKValueTypeDouble
																	   rowCount:rowCount columnCount:1];
	LNKMatrix *trainingSetPolynomial = [[trainingSetSingle polynomialMatrixOfDegree:8] retain];
	LNKMatrix *trainingSetNormalized = trainingSetPolynomial.normalizedMatrix;
	
	LNKMatrix *cvSetSingle = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:xcvURL matrixValueType:LNKValueTypeDouble
														outputVectorAtURL:ycvURL outputVectorValueType:LNKValueTypeDouble
																 rowCount:21 columnCount:1];
	LNKMatrix *cvSetPolynomial = [[cvSetSingle polynomialMatrixOfDegree:8] retain];
	LNKMatrix *cvSetNormalized = [cvSetPolynomial normalizedMatrixWithMeanVector:trainingSetNormalized.normalizationMeanVector standardDeviationVector:trainingSetNormalized.normalizationStandardDeviationVector];
	
	LNKFloat trainError[rowCount];
	LNKFloat cvError[rowCount];
	
	LNKOptimizationAlgorithmLBFGS *cvAlgorithm = [[LNKOptimizationAlgorithmLBFGS alloc] init];
	
	for (LNKSize i = 1; i <= rowCount; i++) {
		[trainingSetNormalized clipRowCountTo:i];
		
		LNKOptimizationAlgorithmLBFGS *algorithm = [[LNKOptimizationAlgorithmLBFGS alloc] init];
		algorithm.lambda = 0;
		
		LNKLinRegPredictor *trainPredictor = [[LNKLinRegPredictor alloc] initWithMatrix:trainingSetNormalized implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:algorithm];
		[trainPredictor train];
		
		// When predicting, we don't want to regularize.
		algorithm.lambda = 0;
		
		trainError[i-1] = [trainPredictor _evaluateCostFunction];
		
		LNKLinRegPredictor *cvPredictor = [[LNKLinRegPredictor alloc] initWithMatrix:cvSetNormalized implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:cvAlgorithm];
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
	
	[trainingSetSingle release];
	[trainingSetPolynomial release];
	[cvSetSingle release];
	[cvSetPolynomial release];
}

- (void)testAICAndBIC {
	NSURL *const mtcarsURL = [[NSBundle bundleForClass:self.class] URLForResource:@"mtcars" withExtension:@"txt"];

	// Only keep mpg and drat
	LNKMatrix *const matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:mtcarsURL delimiter:',' ignoringHeader:YES columnPreprocessingRules:@{
		@0: [LNKCSVColumnRule deleteRule],
		@1: [LNKCSVColumnRule outputRule],
		@2: [LNKCSVColumnRule deleteRule],
		@3: [LNKCSVColumnRule deleteRule],
		@4: [LNKCSVColumnRule deleteRule],
		@6: [LNKCSVColumnRule deleteRule],
		@7: [LNKCSVColumnRule deleteRule],
		@8: [LNKCSVColumnRule deleteRule],
		@9: [LNKCSVColumnRule deleteRule],
		@10: [LNKCSVColumnRule deleteRule],
		@11: [LNKCSVColumnRule deleteRule]
	}];

	id<LNKOptimizationAlgorithm> algorithm = [[LNKOptimizationAlgorithmLBFGS alloc] init];
	LNKLinRegPredictor *const predictor = [[LNKLinRegPredictor alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:algorithm];
	[algorithm release];
	[matrix release];

	[predictor train];

	LNKFloat *const thetaVector = [predictor _thetaVector];
	XCTAssertEqualWithAccuracy(thetaVector[0], -7.525, 0.1);
	XCTAssertEqualWithAccuracy(thetaVector[1],  7.678, 0.1);

	XCTAssertEqualWithAccuracy([predictor computeAIC], 190.7999, 0.1);
	XCTAssertEqualWithAccuracy([predictor computeBIC], 195.1971, 0.1);

	[predictor release];
}

@end
