//
//  NaiveBayesTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>

#import "LNKDiscreteProbabilityDistribution.h"
#import "LNKGaussianProbabilityDistribution.h"
#import "LNKMatrix.h"
#import "LNKNaiveBayesClassifier.h"

@interface NaiveBayesTests : XCTestCase
@end

@implementation NaiveBayesTests

- (void)_registerValuesForDistribution:(LNKDiscreteProbabilityDistribution *)distribution {
	NSParameterAssert(distribution != nil);

	[distribution registerValues:@[ @0, @1 ] forColumnAtIndex:0];
	[distribution registerValues:@[ @0, @1 ] forColumnAtIndex:1];
	[distribution registerValues:@[ @0, @1, @2 ] forColumnAtIndex:2];
	[distribution registerValues:@[ @0, @1 ] forColumnAtIndex:3];
}

- (void)testNaiveBayes {
	// Columns of Flu.csv: chills, runny nose, headache, fever, flu? (output)
	NSURL *const url = [[NSBundle bundleForClass:self.class] URLForResource:@"Flu" withExtension:@"csv"];
	LNKMatrix *const matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:url];
	LNKClasses *const classes = [LNKClasses withCount:2];

	LNKDiscreteProbabilityDistribution *const discreteDistribution = [[LNKDiscreteProbabilityDistribution alloc] initWithClasses:classes featureCount:matrix.columnCount];
	discreteDistribution.performsLaplacianSmoothing = NO;

	LNKNaiveBayesClassifier *const classifier = [[LNKNaiveBayesClassifier alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:nil classes:classes probabilityDistribution:discreteDistribution];
	[matrix release];
	[discreteDistribution release];

	[self _registerValuesForDistribution:discreteDistribution];
	[classifier train];

	const LNKFloat inputVector[] = {1,0,1,0};
	LNKClass *outputClass = [classifier predictValueForFeatureVector:LNKVectorMakeUnsafe(inputVector, 4)];

	XCTAssertEqual(outputClass.unsignedIntegerValue, 0ULL, @"We should not have the flu");

	[classifier release];
}

- (LNKNaiveBayesClassifier *)_classifierForFluChills {
	// Columns of FluChills.csv: chills (always 1), runny nose, headache, fever, flu? (output)
	NSURL *const url = [[NSBundle bundleForClass:self.class] URLForResource:@"FluChills" withExtension:@"csv"];
	LNKMatrix *const matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:url];
	LNKClasses *const classes = [LNKClasses withCount:2];

	LNKDiscreteProbabilityDistribution *const discreteDistribution = [[LNKDiscreteProbabilityDistribution alloc] initWithClasses:classes featureCount:matrix.columnCount];

	LNKNaiveBayesClassifier *const classifier = [[LNKNaiveBayesClassifier alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:nil classes:classes probabilityDistribution:discreteDistribution];
	[matrix release];
	[discreteDistribution release];

	return [classifier autorelease];
}

- (void)testNaiveBayesWithZeroProbability {
	LNKNaiveBayesClassifier *const classifier = [self _classifierForFluChills];

	LNKDiscreteProbabilityDistribution *const probabilityDistribution = classifier.probabilityDistribution;
	probabilityDistribution.performsLaplacianSmoothing = NO;

	[self _registerValuesForDistribution:probabilityDistribution];
	[classifier train];

	const LNKFloat inputVector[] = {0,0,1,0};
	LNKFloat probability = 0;
	LNKClass *const outputClass = [classifier predictValueForFeatureVector:LNKVectorMakeUnsafe(inputVector, 4) probability:&probability];

	XCTAssertEqual(probability, 0, @"There were no prior example of flus without chills");
	XCTAssertNil(outputClass, @"No information");
}

- (void)testNaiveBayesWithLaplacianSmoothing {
	LNKNaiveBayesClassifier *const classifier = [self _classifierForFluChills];

	LNKDiscreteProbabilityDistribution *const probabilityDistribution = classifier.probabilityDistribution;
	probabilityDistribution.performsLaplacianSmoothing = YES;

	[self _registerValuesForDistribution:probabilityDistribution];
	[classifier train];

	const LNKFloat inputVector[] = {0,0,1,0};
	LNKFloat probability = 0;
	LNKClass *const outputClass = [classifier predictValueForFeatureVector:LNKVectorMakeUnsafe(inputVector, 4) probability:&probability];

	XCTAssertGreaterThan(probability, 0);
	XCTAssertNotNil(outputClass);
	XCTAssertEqual(outputClass.unsignedIntegerValue, 0ULL);
}

- (void)testGaussianNaiveBayes {
	NSURL *const url = [[NSBundle bundleForClass:self.class] URLForResource:@"Pima" withExtension:@"csv"];
	LNKMatrix *const matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:url];
	LNKMatrix *trainingMatrix = nil;
	LNKMatrix *testMatrix = nil;
	[matrix splitIntoTrainingMatrix:&trainingMatrix testMatrix:&testMatrix trainingBias:0.8];
	[matrix release];
	
	LNKClasses *const classes = [LNKClasses withCount:2];

	LNKGaussianProbabilityDistribution *const distribution = [[LNKGaussianProbabilityDistribution alloc] initWithClasses:classes featureCount:trainingMatrix.columnCount];

	LNKNaiveBayesClassifier *const classifier = [[LNKNaiveBayesClassifier alloc] initWithMatrix:trainingMatrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:nil classes:classes probabilityDistribution:distribution];
	[distribution release];

	[classifier train];

	const LNKFloat accuracy = [classifier computeClassificationAccuracyOnMatrix:testMatrix];
	XCTAssertGreaterThan(accuracy, 0.65);

	[classifier release];
}

@end
