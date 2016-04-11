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
	NSURL *url = [[NSBundle bundleForClass:[self class]] URLForResource:@"ServerStatistics" withExtension:@"mat"];
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:url matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
															rowCount:307
														 columnCount:2];
	LNKAnomalyDetector *detector = [[LNKAnomalyDetector alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:nil];
	detector.threshold = 8.99e-05;
	
	[matrix release];
	
	[detector train];
	
	LNKClass *class = [detector predictValueForFeatureVector:LNKVectorCreateUnsafe([matrix rowAtIndex:0], matrix.columnCount)];
	XCTAssertEqual(class.unsignedIntegerValue, 0ULL, @"Not an anomaly!");
	
	[detector release];
}

- (void)test2 {
	NSURL *url = [[NSBundle bundleForClass:[self class]] URLForResource:@"ServerStatistics" withExtension:@"mat"];
	NSURL *urlVal = [[NSBundle bundleForClass:[self class]] URLForResource:@"ServerStatisticsVal" withExtension:@"mat"];
	NSURL *urlValY = [[NSBundle bundleForClass:[self class]] URLForResource:@"ServerStatisticsValY" withExtension:@"mat"];
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:url matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
															rowCount:307
														 columnCount:2];
	
	LNKMatrix *cvMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:urlVal matrixValueType:LNKValueTypeDouble
													 outputVectorAtURL:urlValY outputVectorValueType:LNKValueTypeDouble
															  rowCount:307
														   columnCount:2];
	
	LNKFloat threshold = LNKFindAnomalyThreshold(matrix, cvMatrix);
	XCTAssertEqualWithAccuracy(threshold, 8.99e-05, 0.01, @"Incorrect threshold");
	
	[matrix release];
	[cvMatrix release];
}

- (void)test3 {
	NSURL *url = [[NSBundle bundleForClass:[self class]] URLForResource:@"AnomalyPerformance" withExtension:@"mat"];
	NSURL *urlVal = [[NSBundle bundleForClass:[self class]] URLForResource:@"AnomalyPerformanceVal" withExtension:@"mat"];
	NSURL *urlValY = [[NSBundle bundleForClass:[self class]] URLForResource:@"AnomalyPerformanceValY" withExtension:@"mat"];
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:url matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
															rowCount:1000
														 columnCount:11];
	
	LNKMatrix *cvMatrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:urlVal matrixValueType:LNKValueTypeDouble
													 outputVectorAtURL:urlValY outputVectorValueType:LNKValueTypeDouble
															  rowCount:100
														   columnCount:11];
	
	LNKFloat threshold = LNKFindAnomalyThreshold(matrix, cvMatrix);
	XCTAssertEqualWithAccuracy(threshold, 1.38e-18, 0.01, @"Incorrect threshold");
	
	[matrix release];
	[cvMatrix release];
}

@end
