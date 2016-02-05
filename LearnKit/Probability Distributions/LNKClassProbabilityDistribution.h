//
//  LNKClassProbabilityDistribution.h
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class LNKClasses, LNKMatrix;

/// Abstract
/// Models probability distributions for use with LNKClassifiers.
@interface LNKClassProbabilityDistribution : NSObject

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

- (instancetype)initWithClasses:(LNKClasses *)classes featureCount:(LNKSize)featureCount NS_DESIGNATED_INITIALIZER;

@property (nonatomic, retain, nonnull, readonly) LNKClasses *classes;
@property (nonatomic, readonly) LNKSize featureCount;

- (LNKFloat)priorForClassAtIndex:(LNKSize)index;
- (LNKFloat)probabilityLogForClassAtIndex:(LNKSize)classIndex featureAtIndex:(LNKSize)featureIndex value:(LNKFloat)value;

// Subclasses must override this method to build a probability distribution.
- (void)buildWithMatrix:(LNKMatrix *)matrix;

@end

NS_ASSUME_NONNULL_END
