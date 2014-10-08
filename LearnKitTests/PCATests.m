//
//  PCATests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <XCTest/XCTest.h>

#import "LNKMatrix.h"
#import "LNKMatrixPCA.h"

@interface PCATests : XCTestCase

@end

@implementation PCATests

#define DACCURACY 0.01

- (void)test1 {
	NSString *path = [[NSBundle bundleForClass:[self class]] pathForResource:@"ex7data1_X" ofType:@"dat"];
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:path] matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
														exampleCount:50
														 columnCount:2
													addingOnesColumn:NO];
	[matrix normalize];
	
	LNKMatrix *reducedMatrix = [[matrix matrixReducedToDimension:1] retain];
	XCTAssertEqualWithAccuracy(reducedMatrix.matrixBuffer[0],  1.481274, DACCURACY);
	XCTAssertEqualWithAccuracy(reducedMatrix.matrixBuffer[1], -0.912912, DACCURACY);
	XCTAssertEqualWithAccuracy(reducedMatrix.matrixBuffer[2],  1.212087, DACCURACY);
	
	[reducedMatrix release];
	[matrix release];
}

@end
