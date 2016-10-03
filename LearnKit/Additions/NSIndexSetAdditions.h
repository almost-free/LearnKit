//
//  NSIndexSetAdditions.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef void(^NSIndexSetSimpleEnumerator)(NSUInteger);

@interface NSIndexSet (Additions)

+ (NSIndexSet *)withCount:(NSUInteger)count;

- (NSIndexSet *)indexSetByRandomlySamplingTo:(NSUInteger)size;
- (NSIndexSet *)indexSetByRemovingIndex:(NSUInteger)index;

- (void)enumerateAllIndicesUsingBlock:(NSIndexSetSimpleEnumerator)block;

@end

NS_ASSUME_NONNULL_END
