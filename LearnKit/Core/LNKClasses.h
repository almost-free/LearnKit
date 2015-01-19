//
//  LNKClasses.h
//  LearnKit
//
//  Copyright (c) 2015 Matt Rajca. All rights reserved.
//

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
