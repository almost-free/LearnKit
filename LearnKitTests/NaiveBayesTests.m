//
//  NaiveBayesTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>

#import "LNKMatrix.h"
#import "LNKNaiveBayesClassifier.h"

@interface NaiveBayesTests : XCTestCase

@end

@implementation NaiveBayesTests

- (void)_registerValuesForClassifier:(LNKNaiveBayesClassifier *)classifier {
	[classifier registerValues:@[ @0, @1 ] forColumn:0];
	[classifier registerValues:@[ @0, @1 ] forColumn:1];
	[classifier registerValues:@[ @0, @1, @2 ] forColumn:2];
	[classifier registerValues:@[ @0, @1 ] forColumn:3];
}

- (void)testNaiveBayes {
	// Columns of Flu.csv: chills, runny nose, headache, fever, flu? (output)
	NSString *path = [[NSBundle bundleForClass:[self class]] pathForResource:@"Flu" ofType:@"csv"];
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:[NSURL fileURLWithPath:path] addingOnesColumn:NO];

	LNKNaiveBayesClassifier *classifier = [[LNKNaiveBayesClassifier alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:nil classes:[LNKClasses withCount:2]];
	classifier.performsLaplacianSmoothing = NO;
	[matrix release];

	[self _registerValuesForClassifier:classifier];
	[classifier train];

	const LNKFloat inputVector[] = {1,0,1,0};
	LNKClass *outputClass = [classifier predictValueForFeatureVector:LNKVectorMakeUnsafe(inputVector, 4)];

	XCTAssertEqual(outputClass.unsignedIntegerValue, 0ULL, @"We should not have the flu");

	[classifier release];
}

- (LNKNaiveBayesClassifier *)_classifierForFluChills {
	// Columns of FluChills.csv: chills (always 1), runny nose, headache, fever, flu? (output)
	NSString *path = [[NSBundle bundleForClass:[self class]] pathForResource:@"FluChills" ofType:@"csv"];
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:[NSURL fileURLWithPath:path] addingOnesColumn:NO];

	LNKNaiveBayesClassifier *classifier = [[LNKNaiveBayesClassifier alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:nil classes:[LNKClasses withCount:2]];
	[matrix release];

	return [classifier autorelease];
}

- (void)testNaiveBayesWithZeroProbability {
	LNKNaiveBayesClassifier *classifier = [self _classifierForFluChills];
	classifier.performsLaplacianSmoothing = NO;

	[self _registerValuesForClassifier:classifier];
	[classifier train];

	const LNKFloat inputVector[] = {0,0,1,0};
	LNKFloat probability = 0;
	LNKClass *outputClass = [classifier predictValueForFeatureVector:LNKVectorMakeUnsafe(inputVector, 4) probability:&probability];

	XCTAssertEqual(probability, 0, @"There were no prior example of flus without chills");
	XCTAssertNil(outputClass, @"No information");
}

- (void)testNaiveBayesWithLaplacianSmoothing {
	LNKNaiveBayesClassifier *classifier = [self _classifierForFluChills];
	classifier.performsLaplacianSmoothing = YES;

	[self _registerValuesForClassifier:classifier];
	[classifier train];

	const LNKFloat inputVector[] = {0,0,1,0};
	LNKFloat probability = 0;
	LNKClass *outputClass = [classifier predictValueForFeatureVector:LNKVectorMakeUnsafe(inputVector, 4) probability:&probability];

	XCTAssertGreaterThan(probability, 0);
	XCTAssertNotNil(outputClass);
	XCTAssertEqual(outputClass.unsignedIntegerValue, 0ULL);
}

@end
