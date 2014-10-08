//
//  LNKDesignMatrixUI.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKDesignMatrix.h"

@interface LNKDesignMatrix (UI)

#if TARGET_OS_MAC
- (NSImage *)imageForExampleAtIndex:(LNKSize)index width:(NSUInteger)width height:(NSUInteger)height;
#endif

@end
