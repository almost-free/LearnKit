//
//  KMeansTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>

#import "LNKKMeansClassifier.h"
#import "LNKKMeansClassifierPrivate.h"
#import "LNKMatrix.h"

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
	
	LNKVector cluster1 = [classifier centroidForClusterAtIndex:0];
	LNKVector cluster2 = [classifier centroidForClusterAtIndex:1];
	LNKVector cluster3 = [classifier centroidForClusterAtIndex:2];
	
	XCTAssertEqual(cluster1.length, 2UL);
	
	XCTAssertEqualWithAccuracy(cluster1.data[0], 1.9540, DACCURACY);
	XCTAssertEqualWithAccuracy(cluster1.data[1], 5.0256, DACCURACY);
	XCTAssertEqualWithAccuracy(cluster2.data[0], 3.0437, DACCURACY);
	XCTAssertEqualWithAccuracy(cluster2.data[1], 1.0154, DACCURACY);
	XCTAssertEqualWithAccuracy(cluster3.data[0], 6.0337, DACCURACY);
	XCTAssertEqualWithAccuracy(cluster3.data[1], 3.0005, DACCURACY);
	
	LNKVectorFree(cluster1);
	LNKVectorFree(cluster2);
	LNKVectorFree(cluster3);
	
	[classifier release];
}

@end
