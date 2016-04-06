//
//  KNNTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>

#import "LNKKNNClassifier.h"
#import "LNKMatrix.h"

@interface KNNTests : XCTestCase

@end

@implementation KNNTests

- (void)test1 {
	NSString *path = [[NSBundle bundleForClass:[self class]] pathForResource:@"ex2data1" ofType:@"csv"];
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:[NSURL fileURLWithPath:path]];
	
	LNKKNNClassifier *classifier = [[LNKKNNClassifier alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:nil classes:[LNKClasses withCount:2]];
	[matrix release];
	
	for (LNKSize k = 1; k <= 10; k++) {
		classifier.k = k;
		
		LNKFloat inputVector[2] = { 96, 69 };
		XCTAssertEqualObjects([classifier predictValueForFeatureVector:LNKVectorMakeUnsafe(inputVector, 2)], [LNKClass classWithUnsignedInteger:1], @"Incorrect class");
		
		LNKFloat inputVector2[2] = { 49, 50 };
		XCTAssertEqualObjects([classifier predictValueForFeatureVector:LNKVectorMakeUnsafe(inputVector2, 2)], [LNKClass classWithUnsignedInteger:0], @"Incorrect class");
	}
	
	[classifier release];
}

@end
