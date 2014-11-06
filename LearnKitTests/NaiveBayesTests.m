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

- (void)test1 {
	// Columns of Flu.csv: chills, runny nose, headache, fever, flu? (output)
	NSString *path = [[NSBundle bundleForClass:[self class]] pathForResource:@"Flu" ofType:@"csv"];
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:[NSURL fileURLWithPath:path] addingOnesColumn:NO];
	
	LNKNaiveBayesClassifier *classifier = [[LNKNaiveBayesClassifier alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:nil classes:[LNKClasses withCount:2]];
	[matrix release];
	
	[classifier registerValues:@[ @0, @1 ] forColumn:0];
	[classifier registerValues:@[ @0, @1 ] forColumn:1];
	[classifier registerValues:@[ @0, @1, @2 ] forColumn:2];
	[classifier registerValues:@[ @0, @1 ] forColumn:3];
	
	[classifier train];
	
	const LNKFloat inputVector[] = {1,0,1,0};
	LNKClass *outputClass = [classifier predictValueForFeatureVector:LNKVectorMake(inputVector, 4)];
	
	XCTAssertEqual(outputClass.unsignedIntegerValue, 0ULL, @"We should not have the flu");
	
	[classifier release];
}

@end
