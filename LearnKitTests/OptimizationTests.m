//
//  OptimizationTests.m
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "LNKGoldenSectionSearch.h"
#import <XCTest/XCTest.h>

#import "LNKAccelerate.h"

@interface OptimizationTests : XCTestCase
@end

@implementation OptimizationTests

- (void)testGSS
{
	LNKGoldenSectionSearch *gss = [[LNKGoldenSectionSearch alloc] initWithFunction:^LNKFloat(LNKFloat x) {
		return x * x - 2 * x + 1;
	} searchInterval:LNKSearchIntervalMake(0, 2)];
	[gss start];
	XCTAssertEqualWithAccuracy(gss.optimalX, 1.0f, 0.000001);
	[gss release];
}

@end
