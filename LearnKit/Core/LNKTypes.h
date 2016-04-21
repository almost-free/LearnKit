//
//  LNKTypes.h
//  LearnKit
//
//  Copyright (c) 2014 Matt Rajca. All rights reserved.
//

#import "Config.h"
#import <Foundation/Foundation.h>

typedef NS_ENUM(NSUInteger, LNKImplementationType) {
	LNKImplementationTypeAccelerate,
	
#if TARGET_OS_MAC
	LNKImplementationTypeOpenCL,
#endif
	
#if TARGET_OS_IPHONE
	LNKImplementationTypeMetal,
#endif
};


#if USE_DOUBLE_PRECISION
typedef double LNKFloat;
#define LNKFloatMin	-DBL_MAX
#define LNKFloatMax	DBL_MAX
#else
typedef float LNKFloat;
#define LNKFloatMin	-FLT_MAX
#define LNKFloatMax	FLT_MAX
#endif

typedef uint64_t LNKSize;
#define LNKSizeMax UINT64_MAX

#define LNKFloatAlloc(size) (LNKFloat *)malloc((size) * sizeof(LNKFloat))
#define LNKFloatCalloc(size) (LNKFloat *)calloc((size), sizeof(LNKFloat))
#define LNKFloatCopy(dest, src, size) memcpy(dest, src, (size) * sizeof(LNKFloat))

#define LNKFloatAllocAndCopy(data, size) LNKFloatCopy(LNKFloatAlloc(size), data, size)

typedef struct {
	const LNKSize location, length;
} LNKRange;

#define LNKRangeMake(location, length) ((LNKRange) { location, length })

typedef struct {
	LNKSize retainCount;
	const LNKFloat *__nonnull data;
	const LNKSize length;
} LNKVector;

NS_INLINE LNKVector LNKVectorCreate(LNKSize length)
{
	return (LNKVector) { 1, LNKFloatAlloc(length), length };
}

NS_INLINE LNKVector LNKVectorCreateAndCopy(const LNKFloat *__nonnull data, LNKSize length)
{
	return (LNKVector) { 1, LNKFloatAllocAndCopy(data, length), length };
}

NS_INLINE LNKVector LNKVectorCreateUnsafe(const LNKFloat *__nonnull data, LNKSize length)
{
	return (LNKVector) { 1, data, length };
}

NS_INLINE LNKVector LNKVectorWrapUnsafe(const LNKFloat *__nonnull data, LNKSize length)
{
	return (LNKVector) { 0, data, length };
}

NS_INLINE void LNKVectorRetain(LNKVector vector)
{
	vector.retainCount += 1;
}

NS_INLINE void LNKVectorRelease(LNKVector vector)
{
	vector.retainCount -= 1;

	if (vector.retainCount == 0) {
		free((void *)vector.data);
	}
}

@interface NSNumber (LNKTypes)

+ (nonnull NSNumber *)numberWithLNKFloat:(LNKFloat)value;
+ (nonnull NSNumber *)numberWithLNKSize:(LNKSize)value;

@property (nonatomic, readonly) LNKFloat LNKFloatValue;
@property (nonatomic, readonly) LNKSize LNKSizeValue;

@end
