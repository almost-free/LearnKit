//
//  AnomalyDetectorTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKAnomalyDetector.h"
#import "LNKMatrix.h"

#import <Cocoa/Cocoa.h>
#import <XCTest/XCTest.h>

@interface AnomalyDetectorTests : XCTestCase

@end

@implementation AnomalyDetectorTests

- (void)test1 {
	// The first column of the matrix contains network latency numbers.
	// The second column of the matrix contains network throughput numbers.
	NSString *path = [[NSBundle bundleForClass:[self class]] pathForResource:@"ServerStatistics" ofType:@"mat"];
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:path] matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
														exampleCount:307
														 columnCount:2
													addingOnesColumn:NO];
	LNKAnomalyDetector *detector = [[LNKAnomalyDetector alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:nil classes:nil];
	detector.threshold = 8.99e-05;
	
	[matrix release];
	
	[detector train];
	
	LNKClass *class = [detector predictValueForFeatureVector:[matrix exampleAtIndex:0] length:matrix.columnCount];
	XCTAssertEqual(class.unsignedIntegerValue, 0ULL, @"Not an anomaly!");
	
	[detector release];
}

- (void)test2 {
	NSString *path = [[NSBundle bundleForClass:[self class]] pathForResource:@"ServerStatistics" ofType:@"mat"];
	NSString *pathVal = [[NSBundle bundleForClass:[self class]] pathForResource:@"ServerStatisticsVal" ofType:@"mat"];
	NSString *pathValY = [[NSBundle bundleForClass:[self class]] pathForResource:@"ServerStatisticsValY" ofType:@"mat"];
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:path] matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
														exampleCount:307
														 columnCount:2
													addingOnesColumn:NO];
	
	LNKMatrix *cvMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:pathVal] matrixValueType:LNKValueTypeDouble
													 outputVectorAtURL:[NSURL fileURLWithPath:pathValY] outputVectorValueType:LNKValueTypeDouble
														  exampleCount:307
														   columnCount:2
													  addingOnesColumn:NO];
	
	LNKFloat threshold = LNKFindAnomalyThreshold(matrix, cvMatrix);
	XCTAssertEqualWithAccuracy(threshold, 8.99e-05, 0.01, @"Incorrect threshold");
	
	[matrix release];
	[cvMatrix release];
}

@end
