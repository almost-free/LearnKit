//
//  LNKGoldenSectionSearch.h
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

NS_ASSUME_NONNULL_BEGIN

typedef struct {
	LNKFloat start;
	LNKFloat end;
} LNKSearchInterval;

NS_INLINE LNKSearchInterval LNKSearchIntervalMake(LNKFloat start, LNKFloat end)
{
	return (LNKSearchInterval) { start, end };
}

typedef LNKFloat(^LNKUnivariateFunction)(LNKFloat);

/// To run `LNKGoldenSectionSearch` synchronously, call `start`.
///
/// To run `LNKGoldenSectionSearch` asynchronously, enqueue an instance of it onto a `NSOperationQueue`.
/// The `completionBlock` will be called when the operation completes. In it, you can access `optimalX` safely.
@interface LNKGoldenSectionSearch : NSOperation

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

/// Uses a default tolerance of 10^(-8).
/// Note that the function may be called on a secondary thread.
- (instancetype)initWithFunction:(LNKUnivariateFunction)function searchInterval:(LNKSearchInterval)interval;

- (instancetype)initWithFunction:(LNKUnivariateFunction)function searchInterval:(LNKSearchInterval)interval tolerance:(LNKFloat)tolerance;

/// This variable holds the optimal value if a golden section search has been run, and `LNKFloatMax` otherwise.
@property (readonly) LNKFloat optimalX;

@end

NS_ASSUME_NONNULL_END
