//
//  LNKConfusionMatrix.h
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class LNKClass;

@interface LNKConfusionMatrix : NSObject

- (NSUInteger)frequencyForTrueClass:(LNKClass *)trueClass predictedClass:(LNKClass *)predictedClass;

@end

NS_ASSUME_NONNULL_END
