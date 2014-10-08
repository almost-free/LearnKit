//
//  KNNTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>

#import "LNKDesignMatrix.h"
#import "LNKKNNClassifier.h"

@interface KNNTests : XCTestCase

@end

@implementation KNNTests

- (void)test1 {
	NSString *path = [[NSBundle bundleForClass:[self class]] pathForResource:@"ex2data1" ofType:@"csv"];
	LNKDesignMatrix *matrix = [[LNKDesignMatrix alloc] initWithCSVFileAtURL:[NSURL fileURLWithPath:path] addingOnesColumn:NO];
	
	LNKKNNClassifier *classifier = [[LNKKNNClassifier alloc] initWithDesignMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:nil classes:[LNKClasses withCount:2]];
	[matrix release];
	
	for (LNKSize k = 1; k <= 10; k++) {
		classifier.k = k;
		
		LNKFloat inputVector[2] = { 96, 69 };
		XCTAssertEqualObjects([classifier predictValueForFeatureVector:inputVector length:2], [LNKClass classWithUnsignedInteger:1], @"Incorrect class");
		
		LNKFloat inputVector2[2] = { 49, 50 };
		XCTAssertEqualObjects([classifier predictValueForFeatureVector:inputVector2 length:2], [LNKClass classWithUnsignedInteger:0], @"Incorrect class");
	}
	
	[classifier release];
}

@end
