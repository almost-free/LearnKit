//
//  LNKMatrixUI.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#if TARGET_OS_MAC

#import "LNKMatrix.h"

@interface LNKMatrix (UI)

- (NSImage *)imageForExampleAtIndex:(LNKSize)index width:(NSUInteger)width height:(NSUInteger)height;

@end

#endif
