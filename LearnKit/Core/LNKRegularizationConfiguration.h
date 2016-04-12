//
//  LNKRegularizationConfiguration.h
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface LNKRegularizationConfiguration : NSObject

/// The regularization parameter `lambda` must be greater than 0.
- (instancetype)initWithLambda:(LNKFloat)lambda;

+ (instancetype)withLambda:(LNKFloat)lambda;

@property (nonatomic, readonly) LNKFloat lambda;

@end

NS_ASSUME_NONNULL_END
