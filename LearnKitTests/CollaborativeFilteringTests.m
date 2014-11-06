//
//  CollaborativeFilteringTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import <XCTest/XCTest.h>

#import "LNKCollaborativeFilteringPredictorPrivate.h"
#import "LNKMatrix.h"
#import "LNKPredictorPrivate.h"
#import "LNKUtilities.h"

@interface CollaborativeFilteringTests : XCTestCase

@end

@implementation CollaborativeFilteringTests

- (void)testCostFunction {
	const LNKSize movieCount = 1682;
	const LNKSize userCount = 943;
	const LNKSize exampleCount = 10;
	
	NSBundle *bundle = [NSBundle bundleForClass:[self class]];
	NSString *pathX = [bundle pathForResource:@"Movies_X" ofType:@"mat"];
	NSString *pathY = [bundle pathForResource:@"Movies_Y" ofType:@"mat"];
	NSString *pathR = [bundle pathForResource:@"Movies_R" ofType:@"mat"];
	NSString *pathTheta = [bundle pathForResource:@"Movies_Theta" ofType:@"mat"];
	
	NSData *thetaVectorData = LNKLoadBinaryMatrixFromFileAtURL([NSURL fileURLWithPath:pathTheta], userCount * exampleCount * sizeof(double));
	NSData *rMatrixData = LNKLoadBinaryMatrixFromFileAtURL([NSURL fileURLWithPath:pathR], movieCount * userCount * sizeof(double));
	NSData *yMatrixData = LNKLoadBinaryMatrixFromFileAtURL([NSURL fileURLWithPath:pathY], movieCount * userCount * sizeof(double));
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithBinaryMatrixAtURL:[NSURL fileURLWithPath:pathX] matrixValueType:LNKValueTypeDouble
												   outputVectorAtURL:nil outputVectorValueType:LNKValueTypeNone
														exampleCount:movieCount columnCount:exampleCount
													addingOnesColumn:NO];
	
	LNKCollaborativeFilteringPredictor *predictor = [[LNKCollaborativeFilteringPredictor alloc] initWithMatrix:matrix implementationType:LNKImplementationTypeAccelerate optimizationAlgorithm:nil userCount:userCount];
	[predictor _copyThetaVector:(const LNKFloat *)thetaVectorData.bytes shouldTranspose:YES];
	
#warning TODO: these should take a LNKMatrix
	[predictor copyIndicatorMatrix:(const LNKFloat *)rMatrixData.bytes shouldTranspose:YES];
	[predictor copyOutputMatrix:(const LNKFloat *)yMatrixData.bytes shouldTranspose:YES];
	
	XCTAssertEqualWithAccuracy([predictor _evaluateCostFunction], 27918, 1);
}

@end
