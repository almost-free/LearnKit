//
//  LNKDecisionTree.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKClassifier.h"

NS_ASSUME_NONNULL_BEGIN

@interface LNKDecisionTreeNode : NSObject

@end


@interface LNKDecisionTreeClassificationNode : LNKDecisionTreeNode

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)withClass:(LNKClass *)class;
+ (instancetype)unknownClass;

@property (nonatomic, nullable, readonly) LNKClass *classification;

@end


@interface LNKDecisionTreeSplitNode : LNKDecisionTreeNode

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithColumnIndex:(LNKSize)columnIndex;

@property (nonatomic, readonly) LNKSize columnIndex;

- (void)addBranch:(LNKDecisionTreeNode *)branch value:(LNKSize)value;

- (nullable LNKDecisionTreeNode *)branchForValue:(LNKSize)value;

@end

NS_ASSUME_NONNULL_END
