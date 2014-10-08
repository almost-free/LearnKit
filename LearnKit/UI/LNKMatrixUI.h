//
//  LNKMatrixUI.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKMatrix.h"

@interface LNKMatrix (UI)

#if TARGET_OS_MAC
- (NSImage *)imageForExampleAtIndex:(LNKSize)index width:(NSUInteger)width height:(NSUInteger)height;
#endif

@end
