//
//  LNKDecisionTree.m
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKDecisionTree.h"

@implementation LNKDecisionTreeNode

@end


@implementation LNKDecisionTreeClassificationNode

+ (instancetype)withClass:(LNKClass *)class {
	NSParameterAssert(class);
	
	return [[[self alloc] initWithClass:class] autorelease];
}

+ (instancetype)unknownClass {
	return [[[self alloc] initWithClass:nil] autorelease];
}

- (instancetype)initWithClass:(nullable LNKClass *)class {
	self = [super init];
	if (self) {
		_classification = [class retain];
	}
	return self;
}

- (void)dealloc {
	[_classification release];
	[super dealloc];
}

@end



@implementation LNKDecisionTreeSplitNode {
	NSMutableDictionary *_branches;
}

- (instancetype)initWithColumnIndex:(LNKSize)columnIndex {
	NSParameterAssert(columnIndex != LNKSizeMax);
	
	self = [super init];
	if (self) {
		_columnIndex = columnIndex;
		_branches = [[NSMutableDictionary alloc] init];
	}
	return self;
}

- (void)dealloc {
	[_branches release];
	[super dealloc];
}

- (void)addBranch:(LNKDecisionTreeNode *)branch value:(LNKSize)value {
	NSParameterAssert(branch);
	
	_branches[@(value)] = branch;
}

- (LNKDecisionTreeNode *)branchForValue:(LNKSize)value {
	return _branches[@(value)];
}

@end
