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
#import "LNKMatrixImages.h"

@interface KMeansTests : XCTestCase

@end

@implementation KMeansTests

#define DACCURACY 0.01

- (void)test1 {
	NSURL *const url = [[NSBundle bundleForClass:self.class] URLForResource:@"ex7data2_X" withExtension:@"dat"];
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:url matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
															rowCount:300
														 columnCount:2];
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

- (void)testConvergenceChecking {
	NSURL *const url = [[NSBundle bundleForClass:self.class] URLForResource:@"ex7data2_X" withExtension:@"dat"];
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:url matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
															rowCount:300
														 columnCount:2];
	LNKKMeansClassifier *const classifier = [[LNKKMeansClassifier alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:nil classes:[LNKClasses withCount:3]];
	classifier.iterationCount = LNKSizeMax;

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

- (void)testPrimaryColorDetection {
	// Image courtesy of Michael Wifall
	// https://flic.kr/p/eTff8o
	NSURL *const url = [[NSBundle bundleForClass:self.class] URLForResource:@"Palace" withExtension:@"jpg"];
	LNKMatrix *const matrix = [[LNKMatrix alloc] initWithImageAtURL:url format:LNKImageFormatRGB];

	LNKKMeansClassifier *const classifier = [[LNKKMeansClassifier alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:nil classes:[LNKClasses withCount:2]];
	classifier.iterationCount = LNKSizeMax;
	[matrix release];

	[classifier train];

	const LNKVector color1 = [classifier centroidForClusterAtIndex:0];
	const LNKVector color2 = [classifier centroidForClusterAtIndex:1];

	void(^testFirstColor)(LNKVector) = ^(LNKVector v) {
		XCTAssertEqualWithAccuracy(v.data[0], 0.15, DACCURACY);
		XCTAssertEqualWithAccuracy(v.data[1], 0.27, DACCURACY);
		XCTAssertEqualWithAccuracy(v.data[2], 0.444, DACCURACY);
	};

	void(^testSecondColor)(LNKVector) = ^(LNKVector v) {
		XCTAssertEqualWithAccuracy(v.data[0], 0.821, DACCURACY);
		XCTAssertEqualWithAccuracy(v.data[1], 0.739, DACCURACY);
		XCTAssertEqualWithAccuracy(v.data[2], 0.574, DACCURACY);
	};

	if (color1.data[0] < color2.data[0]) {
		testFirstColor(color1);
		testSecondColor(color2);
	} else {
		testFirstColor(color2);
		testSecondColor(color1);
	}

	[classifier release];
}

@end
