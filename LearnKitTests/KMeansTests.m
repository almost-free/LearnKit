//
//  KMeansTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>

#import "LNKMatrix.h"
#import "LNKKMeansClassifier.h"
#import "LNKKMeansClassifierPrivate.h"

@interface KMeansTests : XCTestCase

@end

@implementation KMeansTests

#define DACCURACY 0.01

- (void)test1 {
	NSString *path = [[NSBundle bundleForClass:[self class]] pathForResource:@"ex7data2_X" ofType:@"dat"];
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:path] matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
														exampleCount:300
														 columnCount:2
													addingOnesColumn:NO];
	LNKKMeansClassifier *classifier = [[LNKKMeansClassifier alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:nil classes:[LNKClasses withCount:3]];
	classifier.iterationCount = 10;
	
	LNKFloat clusterCentroids[6];
	clusterCentroids[0] = 3;
	clusterCentroids[1] = 3;
	
	clusterCentroids[2] = 6;
	clusterCentroids[3] = 2;
	
	clusterCentroids[4] = 8;
	clusterCentroids[5] = 5;
	
	[classifier _setClusterCentroids:(const LNKFloat *)clusterCentroids];
	[matrix release];
	
	[classifier train];
	
	XCTAssertEqualWithAccuracy([classifier _clusterCentroids][0], 1.9540, DACCURACY);
	XCTAssertEqualWithAccuracy([classifier _clusterCentroids][1], 5.0256, DACCURACY);
	XCTAssertEqualWithAccuracy([classifier _clusterCentroids][2], 3.0437, DACCURACY);
	XCTAssertEqualWithAccuracy([classifier _clusterCentroids][3], 1.0154, DACCURACY);
	XCTAssertEqualWithAccuracy([classifier _clusterCentroids][4], 6.0337, DACCURACY);
	XCTAssertEqualWithAccuracy([classifier _clusterCentroids][5], 3.0005, DACCURACY);
	
	[classifier release];
}

@end
