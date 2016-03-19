//
//  LNKTopicModeller.h
//  LearnKit
//
//  Created by Matt on 3/19/16.
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class LNKMatrix, LNKTopicModeller;

@interface LNKTopicSet : NSObject
@property (nonatomic, readonly) NSArray<NSArray<NSString *> *> *topics;
@end

@protocol LNKTopicModellerDelegate <NSObject>
- (void)topicModeller:(LNKTopicModeller *)modeller didFindTopics:(LNKTopicSet *)topics;
@end

@interface LNKTopicModeller : NSOperation

/// The document matrix contains documents as row vectors represented as word counts for words in the vocabulary.
- (instancetype)initWithDocumentMatrix:(LNKMatrix *)documentMatrix vocabulary:(NSArray<NSString *> *)vocabulary topicCount:(LNKSize)topicCount delegate:(id<LNKTopicModellerDelegate>)delegate;

@end

NS_ASSUME_NONNULL_END
