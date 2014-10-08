//
//  StructureTests.m
//  LearnKit Tests
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import <XCTest/XCTest.h>

#import "LNKFastFloatQueue.h"

@interface StructureTests : XCTestCase

@end

@implementation StructureTests

- (void)testFastFloatQueue {
	LNKFastFloatQueueRef queue = LNKFastFloatQueueCreate(4);
	
	LNKFastFloatQueueEnqueue(queue, 1);
	LNKFastFloatQueueEnqueue(queue, 2);
	LNKFastFloatQueueEnqueue(queue, 3);
	
	XCTAssertEqual(LNKFastFloatQueueDequeue(queue), 1);
	LNKFastFloatQueueEnqueue(queue, 4);
	LNKFastFloatQueueEnqueue(queue, 5);
	
	LNKFastFloatQueueFree(queue);
}

@end
