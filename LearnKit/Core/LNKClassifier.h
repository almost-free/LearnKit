//
//  LNKClassifier.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "LNKPredictor.h"

@interface LNKClass : NSObject

- (instancetype)init NS_UNAVAILABLE;

+ (instancetype)classWithUnsignedInteger:(NSUInteger)value;

@property (nonatomic, readonly) NSUInteger unsignedIntegerValue;

@end


typedef NSUInteger(^LNKClassMapper)(LNKClass *aClass);

@interface LNKClasses : NSObject <NSFastEnumeration>

/// All integers in the given range are mapped to indices 0..length
+ (instancetype)withRange:(NSRange)range;
+ (instancetype)withCount:(LNKSize)count;
+ (instancetype)withClasses:(NSArray *)classes mapper:(LNKClassMapper)mapper;

- (instancetype)init NS_UNAVAILABLE;

- (NSUInteger)indexForClass:(LNKClass *)aClass;

@property (nonatomic, readonly) NSUInteger count;

@end


/// Abstract
@interface LNKClassifier : LNKPredictor

/// The desired output classes must be non-`nil`.
- (instancetype)initWithMatrix:(LNKMatrix *)matrix implementationType:(LNKImplementationType)implementation optimizationAlgorithm:(id<LNKOptimizationAlgorithm>)algorithm classes:(LNKClasses *)classes;

@property (nonatomic, retain, readonly) LNKClasses *classes;

/// The classifier should be trained prior to calling this method.
- (LNKFloat)computeClassificationAccuracy;

@end
