//
//  LNKOptimization.h
//  LearnKit
//
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

typedef struct {
	LNKFloat start;
	LNKFloat end;
} LNKSearchInterval;

NS_INLINE LNKSearchInterval LNKSearchIntervalMake(LNKFloat start, LNKFloat end)
{
	return (LNKSearchInterval) { start, end };
}

typedef LNKFloat(^LNKUnivariateFunction)(LNKFloat);
typedef LNKFloat(^LNKMultivariateFunction)(LNKVector);
