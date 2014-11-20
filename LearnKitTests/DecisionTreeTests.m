//
//  DecisionTreeTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import <XCTest/XCTest.h>

#import "LNKDecisionTreeClassifier.h"
#import "LNKMatrix.h"

@interface DecisionTreeTests : XCTestCase

@end

@implementation DecisionTreeTests

- (void)testWaitTimesSet {
	NSBundle *bundle = [NSBundle bundleForClass:self.class];
	NSURL *matrixURL = [bundle URLForResource:@"WillWait" withExtension:@"csv"];
	
	LNKMatrix *matrix = [[LNKMatrix alloc] initWithCSVFileAtURL:matrixURL addingOnesColumn:NO];
	
	LNKDecisionTreeClassifier *classifier = [[LNKDecisionTreeClassifier alloc] initWithMatrix:matrix
																		   implementationType:LNKImplementationTypeAccelerate
																		optimizationAlgorithm:nil
																					  classes:[LNKClasses withCount:2]]; /* binary */
	[matrix release];
	
	[classifier registerBooleanValueForColumnAtIndex:0];
	[classifier registerBooleanValueForColumnAtIndex:1];
	[classifier registerBooleanValueForColumnAtIndex:2];
	[classifier registerBooleanValueForColumnAtIndex:3];
	[classifier registerCategoricalValues:3 forColumnAtIndex:4];
	[classifier registerCategoricalValues:3 forColumnAtIndex:5];
	[classifier registerBooleanValueForColumnAtIndex:6];
	[classifier registerBooleanValueForColumnAtIndex:7];
	[classifier registerCategoricalValues:4 forColumnAtIndex:8];
	[classifier registerCategoricalValues:4 forColumnAtIndex:9];
	
	[classifier train];
	
	LNKFloat accuracy = [classifier computeClassificationAccuracy];
	XCTAssertEqualWithAccuracy(accuracy, 1.0, 0.01);
	
	[classifier release];
}

@end
