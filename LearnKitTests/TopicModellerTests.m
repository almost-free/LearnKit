//
//  TopicModellerTests.m
//  LearnKit
//
//  Created by Matt on 3/19/16.
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import <XCTest/XCTest.h>

#import "LNKMatrix.h"
#import "LNKTopicModeller.h"

@interface TopicModellerTests : XCTestCase <LNKTopicModellerDelegate>
@end

@implementation TopicModellerTests {
	XCTestExpectation *_expectation;
}

- (void)dealloc {
	[_expectation release];
	[super dealloc];
}

- (void)testNIPS {
	const LNKSize documentCount = 1500;
	const LNKSize wordCount = 12419;

	NSURL *const docURL = [[NSBundle bundleForClass:self.class] URLForResource:@"nips-docword" withExtension:@"txt"];
	NSURL *const vocabURL = [[NSBundle bundleForClass:self.class] URLForResource:@"nips-vocab" withExtension:@"txt"];

	XCTAssertNotNil(docURL);
	XCTAssertNotNil(vocabURL);

	NSError *error = nil;
	NSString *const document = [[NSString alloc] initWithContentsOfURL:docURL encoding:NSUTF8StringEncoding error:&error];

	LNKMatrix *const matrix = [[LNKMatrix alloc] initWithRowCount:documentCount columnCount:wordCount addingOnesColumn:NO prepareBuffers:^BOOL(LNKFloat *matrixBuffer, LNKFloat *outputVector) {
#pragma unused(outputVector)
		[document enumerateLinesUsingBlock:^(NSString *line, BOOL *stop) {
#pragma unused(stop)
			NSArray<NSString *> *const components = [line componentsSeparatedByString:@" "];

			const int docIndex = [components[0] intValue] - 1; // indices start at 1 in the dataset
			const int wordIndex = [components[1] intValue] - 1;
			const int count = [components[2] intValue];
			matrixBuffer[docIndex * wordCount + wordIndex] = (double)count;
		}];

		return YES;
	}];

	[document release];

	NSError *error2 = nil;
	NSString *const vocabFile = [[NSString alloc] initWithContentsOfURL:vocabURL encoding:NSUTF8StringEncoding error:&error2];

	NSMutableArray<NSString *> *const vocabulary = [[NSMutableArray alloc] init];
	[vocabFile enumerateLinesUsingBlock:^(NSString *line, BOOL *stop) {
#pragma unused(stop)
		[vocabulary addObject:line];
	}];
	[vocabFile release];

	NSOperationQueue *const queue = [[NSOperationQueue alloc] init];
	LNKTopicModeller *const modeller = [[LNKTopicModeller alloc] initWithDocumentMatrix:matrix vocabulary:vocabulary topicCount:30 delegate:self];
	[vocabulary release];
	[matrix release];

	_expectation = [[self expectationWithDescription:@"Expectation"] retain];
	[queue addOperation:modeller];
	[modeller release];

	[self waitForExpectationsWithTimeout:600 handler:nil];
	[queue release];
}

- (void)topicModeller:(LNKTopicModeller *)modeller didFindTopics:(LNKTopicSet *)topics {
#pragma unused(modeller)
	for (NSArray<NSString *> *topic in topics.topics) {
		if ([topic containsObject:@"network"]) {
			[_expectation fulfill];
			return;
		}

		if ([topic containsObject:@"neural"]) {
			[_expectation fulfill];
			return;
		}
	}

	XCTFail();

	[_expectation fulfill];
}

@end
