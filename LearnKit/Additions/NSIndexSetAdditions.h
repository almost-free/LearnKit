//
//  NSIndexSetAdditions.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>

typedef void(^NSIndexSetSimpleEnumerator)(NSUInteger);

@interface NSIndexSet (Additions)

- (NSIndexSet *)indexSetByRemovingIndex:(NSUInteger)index;

- (void)enumerateAllIndicesUsingBlock:(NSIndexSetSimpleEnumerator)block;

@end
