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
	[matrix release];
	
	[detector train];
	
	[detector release];
}

@end
