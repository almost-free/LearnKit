//
//  LNKCSVColumnRule.h
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, LNKCSVColumnRuleType) {
	LNKCSVColumnRuleTypeDelete,
	LNKCSVColumnRuleTypeConversion,
	LNKCSVColumnRuleTypeOutput
};

typedef LNKFloat(^LNKCSVColumnRuleTypeConversionHandler)(NSString *_Nullable);

@interface LNKCSVColumnRule : NSObject

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

+ (instancetype)deleteRule;
+ (instancetype)conversionRuleWithBlock:(LNKCSVColumnRuleTypeConversionHandler)block;
+ (instancetype)outputRule;

@property (nonatomic, readonly) LNKCSVColumnRuleType type;
@property (nonatomic, copy, readonly) id object;

@end

NS_ASSUME_NONNULL_END
